"""Command-line entry points; data, artifacts, and GPU selection are explicit."""
import argparse
import json

from . import workflow
from .policy import StandaloneController


def demo():
    controller = StandaloneController(2., .5)
    trace = [[-.2, -.2, -.2]] * 18 + [[-1., 0., 1.]] * 3 + [[2., 2., 2.]] * 5
    events = []
    for scores in trace:
        decision = controller.step(scores)
        events.append(dict(scores=scores, decision=decision))
        if decision['alarm']:
            break
    return dict(synthetic=True, acceptance_result=False, events=events)


def _assets(parser):
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--directions', required=True)
    parser.add_argument('--generator-spec', required=True)


def _generation(parser):
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--image-root', required=True)
    parser.add_argument('--device', required=True, help='Explicit CUDA device, for example cuda:0')
    parser.add_argument('--memory-limit-gib', type=float, default=20.)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('demo', help='CPU-only synthetic gate demonstration; no benchmark accuracy')
    fit = commands.add_parser('fit', help='Fit three projections from response-prefix-labelled fit vectors')
    fit.add_argument('--inputs', required=True)
    fit.add_argument('--records', required=True)
    fit.add_argument('--features', required=True)
    fit.add_argument('--generator-spec', required=True)
    fit.add_argument('--output', required=True)
    bind = commands.add_parser('import-directions', help='Bind a separately audited fit-only artifact; no refitting')
    _assets(bind)
    bind.add_argument('--output', required=True)
    collect = commands.add_parser('collect', help='Native-greedy smoke and complete calibration-only shadow traces')
    _assets(collect)
    _generation(collect)
    collect.add_argument('--run', required=True)
    collect.add_argument('--max-tokens', type=int, default=128)
    freeze = commands.add_parser('freeze', help='Select only on calibration and freeze before test')
    _assets(freeze)
    freeze.add_argument('--run', required=True)
    freeze.add_argument('--inherited-threshold', type=float)
    test = commands.add_parser('test', help='Generate only test requests with the frozen policy')
    _assets(test)
    _generation(test)
    test.add_argument('--run', required=True)
    report = commands.add_parser('report', help='CPU-only causal replay and complete/partial coverage report')
    _assets(report)
    report.add_argument('--run', required=True)
    report.add_argument('--output', required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop('command')
    function = {'demo': demo, 'fit': workflow.fit, 'import-directions': workflow.bind_directions,
                'collect': workflow.collect, 'freeze': workflow.freeze,
                'test': workflow.test, 'report': workflow.report}[command]
    try:
        result = function(**args)
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
        parser.exit(2, type(error).__name__ + ': ' + str(error) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))
