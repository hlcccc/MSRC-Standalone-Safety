"""Fit/calibration/test separation and causal replay for the portable package."""
from pathlib import Path
import time

from .assets import (context, fitted_model, load_inputs, read_json, read_jsonl,
                     record_path, save_new, sha256)
from .policy import Evidence, StandaloneController, metrics, select_request_threshold
from .signals import fit_directions


def fit(inputs, records, features, generator_spec, output):
    import numpy as np

    rows, _ = load_inputs(inputs)
    checkpoints = read_jsonl(records)
    for point in checkpoints:
        source = rows.get(point.get('id'))
        if not source or source['role'] != 'fit' or point.get('group') != source['group']:
            raise ValueError('fit checkpoint is outside the frozen fit partition')
    with np.load(features, allow_pickle=False) as arrays:
        model = fit_directions(checkpoints, arrays['layers'])
    model = {key: value.tolist() if isinstance(value, np.ndarray) else value
             for key, value in model.items()}
    model.update(partition_sha256=sha256(inputs), generator_spec_sha256=sha256(generator_spec),
                 checkpoint_records_sha256=sha256(records), features_sha256=sha256(features),
                 reused_directions=False)
    save_new(output, model)
    return dict(fit_requests=model['fit_requests'], fit_checkpoints=model['fit_checkpoints'])


def bind_directions(inputs, directions, generator_spec, output):
    """Explicitly import an already-audited fit-only artifact, never silently refit."""
    load_inputs(inputs)
    original = fitted_model(directions)
    fields = ('weight', 'bias', 'hold_threshold', 'fit_groups', 'fit_requests',
              'fit_checkpoints', 'fitting_role', 'shrinkage', 'epsilon')
    model = {key: original[key] for key in fields if key in original}
    model.update(partition_sha256=sha256(inputs), generator_spec_sha256=sha256(generator_spec),
                 imported_directions_sha256=sha256(directions), reused_directions=True,
                 source_fit_audit_required=True)
    save_new(output, model)
    return dict(reused_directions=True, source_fit_audit_required=True)


def identity(row):
    return {key: row[key] for key in ('id', 'group', 'role', 'label', 'image_sha256')}


def audit_record(record, row, expected_context, thresholds=None, hold_threshold=None):
    if record.get('request') != identity(row) or record.get('context') != expected_context:
        raise ValueError('request or generation context changed')
    if record.get('plugguard_used') is not False:
        raise ValueError('record must explicitly exclude PlugGuard')
    trace, tokens = record['layer_score_trace'], record['token_ids']
    if not trace or len(trace) != len(tokens) or len(trace) > expected_context['max_tokens']:
        raise ValueError('one score vector per generated non-EOS token required')
    if any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError('nonnegative integer token IDs required')
    started, ended = record['start_ns'], record['end_ns']
    if type(started) is not int or type(ended) is not int or not 0 < started <= ended:
        raise ValueError('ordered generation timestamps required')
    if record['stop_reason'] not in ('eos', 'max_tokens', 'all_arms_blocked'):
        raise ValueError('unknown generation stop reason')
    if record['stop_reason'] == 'max_tokens' and len(trace) != expected_context['max_tokens']:
        raise ValueError('incomplete max-token trace')
    evidence = Evidence()
    for scores in trace:
        evidence.step(scores)
    if thresholds is None:
        if (record.get('arms') != {} or record.get('full_shadow') is not True or
                record['stop_reason'] == 'all_arms_blocked'):
            raise ValueError('full, unfiltered calibration shadow required')
        return
    if record.get('full_shadow') is not False or set(record['arms']) != set(thresholds):
        raise ValueError('test arms or full-shadow mode differ')
    blocked = []
    for name, threshold in thresholds.items():
        arm = record['arms'][name]
        controller = StandaloneController(threshold, hold_threshold)
        decision, last_time = None, started
        for position, scores in enumerate(trace, 1):
            value = controller.step(scores)
            if position > len(arm['events']):
                raise ValueError('missing online scoring event')
            event = arm['events'][position - 1]
            stamp = event['score_ns']
            if (type(stamp) is not int or not last_time <= stamp <= ended or
                    event['position'] != position or event['decision'] != value):
                raise ValueError('causal decision replay or scoring timestamp differs')
            last_time = stamp
            if value['alarm']:
                decision = value
                break
        if decision is None:
            decision = controller.finish()
        if len(arm['events']) != decision['generated'] or arm['decision'] != decision:
            raise ValueError('terminal decision or event coverage differs')
        blocked.append(decision['alarm'])
    if (record['stop_reason'] == 'all_arms_blocked') != all(blocked):
        # An EOS/limit at the same token as the last alarm is recorded as blocked.
        raise ValueError('stop reason differs from gate decisions')


def _decoder(factory, model_dir, image_root, generator_spec, directions, device, memory_limit_gib):
    if factory is None:
        from .generation import Decoder
        factory = Decoder
    return factory(model_dir, image_root, generator_spec, fitted_model(directions), device, memory_limit_gib)


def _records(run, role, rows, roles, expected_context, thresholds=None, hold_threshold=None):
    records, hashes = {}, {}
    for path in sorted((Path(run) / role).glob('*.json')):
        record = read_json(path)
        rid = record['request']['id']
        if rid in records or rid not in roles[role] or path.resolve() != record_path(run, role, rid):
            raise ValueError('unexpected, duplicate, or misnamed request record')
        audit_record(record, rows[rid], expected_context, thresholds, hold_threshold)
        records[rid] = record
        hashes[path.name] = sha256(path)
    return records, hashes


def collect(inputs, directions, generator_spec, run, model_dir, image_root, device,
            memory_limit_gib=20, max_tokens=128, decoder_factory=None):
    run = Path(run)
    if (run / 'protocol.json').exists() or (run / 'test').exists():
        raise ValueError('cannot recollect calibration after freezing or test start')
    rows, roles = load_inputs(inputs)
    expected = context(inputs, directions, generator_spec, max_tokens)
    metadata = run / 'development.json'
    if metadata.exists():
        if read_json(metadata) != expected:
            raise ValueError('development generation context changed')
    else:
        save_new(metadata, expected)
    records, _ = _records(run, 'calibration', rows, roles, expected)
    if set(records) == roles['calibration']:
        return dict(completed=len(records), resumed=True)
    decoder = _decoder(decoder_factory, model_dir, image_root, generator_spec,
                       directions, device, memory_limit_gib)
    checks = decoder.smoke([rows[rid] for rid in sorted(roles['calibration'])])
    smoke = run / 'smoke.json'
    if smoke.exists():
        if read_json(smoke) != checks:
            raise ValueError('native-greedy smoke changed')
    else:
        save_new(smoke, checks)
    for rid in sorted(roles['calibration'] - set(records)):
        result = decoder.decode(rows[rid], {}, max_tokens=max_tokens, full_shadow=True)
        result.update(request=identity(rows[rid]), context=expected)
        audit_record(result, rows[rid], expected)
        save_new(record_path(run, 'calibration', rid), result)
    return dict(completed=len(roles['calibration']), calibration_only=True)


def freeze(inputs, directions, generator_spec, run, inherited_threshold=None):
    run = Path(run)
    if (run / 'test').exists() or (run / 'protocol.json').exists():
        raise ValueError('cannot select again after freezing or test start')
    rows, roles = load_inputs(inputs)
    expected = read_json(run / 'development.json')
    if context(inputs, directions, generator_spec, expected['max_tokens']) != expected:
        raise ValueError('development artifacts changed')
    smoke = read_json(run / 'smoke.json')
    suites = {rows[rid].get('suite', 'unspecified') for rid in roles['calibration']}
    if (smoke.get('calibration_only') is not True or smoke.get('no_online_guard_model') is not True or
            {check['suite'] for check in smoke['checks']} != suites or
            not all(check['exact_greedy_match'] is True for check in smoke['checks'])):
        raise ValueError('native-greedy calibration smoke required')
    records, hashes = _records(run, 'calibration', rows, roles, expected)
    if set(records) != roles['calibration']:
        raise ValueError('all calibration requests required before threshold selection')
    maxima = []
    for rid, record in records.items():
        evidence = Evidence()
        maximum = max(evidence.step(scores)['temporal'] for scores in record['layer_score_trace'])
        maxima.append(dict(id=rid, group=rows[rid]['group'], role='calibration',
                           label=rows[rid]['label'], score=maximum))
    selection = select_request_threshold(maxima)
    arms = dict(standalone_recalibrated=selection['threshold'])
    hold = fitted_model(directions)['hold_threshold']
    if inherited_threshold is not None:
        StandaloneController(inherited_threshold, hold)
        arms['standalone_inherited'] = inherited_threshold
    protocol = dict(schema_version=1, context=expected, arms=arms, hold_threshold=hold,
                    buffers=[16, 64], calibration_selection=selection,
                    calibration_records_sha256=hashes, smoke_sha256=sha256(run / 'smoke.json'),
                    roles={key: len(value) for key, value in roles.items()},
                    generation=dict(dtype='bfloat16', attention='sdpa', max_pixels=1003520,
                                    greedy=True, seed=20261024),
                    plugguard_used=False, entropy_gate_used=False,
                    test_used_for_fitting_or_selection=False, frozen_ns=time.time_ns())
    save_new(run / 'protocol.json', protocol)
    save_new(run / 'frozen.json', dict(protocol_sha256=sha256(run / 'protocol.json')))
    return dict(thresholds=arms, calibration=selection['calibration'])


def verify(inputs, directions, generator_spec, run):
    run = Path(run)
    if sha256(run / 'protocol.json') != read_json(run / 'frozen.json')['protocol_sha256']:
        raise ValueError('frozen protocol changed')
    protocol = read_json(run / 'protocol.json')
    if context(inputs, directions, generator_spec, protocol['context']['max_tokens']) != protocol['context']:
        raise ValueError('frozen sources, partition, or directions changed')
    rows, roles = load_inputs(inputs)
    records, hashes = _records(run, 'calibration', rows, roles, protocol['context'])
    if set(records) != roles['calibration'] or hashes != protocol['calibration_records_sha256']:
        raise ValueError('frozen calibration coverage or bytes changed')
    if sha256(run / 'smoke.json') != protocol['smoke_sha256']:
        raise ValueError('frozen smoke check changed')
    return protocol, rows, roles


def test(inputs, directions, generator_spec, run, model_dir, image_root, device,
         memory_limit_gib=20, decoder_factory=None):
    protocol, rows, roles = verify(inputs, directions, generator_spec, run)
    records, _ = _records(run, 'test', rows, roles, protocol['context'],
                          protocol['arms'], protocol['hold_threshold'])
    todo = sorted(roles['test'] - set(records))
    if not todo:
        return dict(completed=len(records), resumed=True)
    decoder = _decoder(decoder_factory, model_dir, image_root, generator_spec,
                       directions, device, memory_limit_gib)
    for index, rid in enumerate(todo, 1):
        result = decoder.decode(rows[rid], protocol['arms'], max_tokens=protocol['context']['max_tokens'])
        result.update(request=identity(rows[rid]), context=protocol['context'])
        audit_record(result, rows[rid], protocol['context'], protocol['arms'], protocol['hold_threshold'])
        save_new(record_path(run, 'test', rid), result)
        if index % 10 == 0:
            print('Verified test requests: ' + str(len(records) + index), flush=True)
    return dict(completed=len(roles['test']), test_only=True)


def report(inputs, directions, generator_spec, run, output):
    protocol, rows, roles = verify(inputs, directions, generator_spec, run)
    records, _ = _records(run, 'test', rows, roles, protocol['context'],
                          protocol['arms'], protocol['hold_threshold'])
    primary = {name: metrics([dict(label=rows[rid]['label'], warn=record['arms'][name]['decision']['alarm'])
                              for rid, record in records.items()]) for name in protocol['arms']}
    result = dict(status='COMPLETE' if set(records) == roles['test'] else 'PARTIAL_NOT_FINAL_ACCURACY',
                  completed=len(records), required=len(roles['test']), primary=primary,
                  thresholds=protocol['arms'], primary_metric='request_label_warning_accuracy',
                  causal_replay_verified_records=len(records), plugguard_used=False,
                  online_per_token_scoring=True, decision_before_buffer_release=True,
                  joint_shadow_for_two_thresholds=len(protocol['arms']) > 1,
                  no_independent_arm_latency_claim=True,
                  harmful_token_early_warning_accuracy_not_proven=True,
                  answer_safety_not_judged_in_this_ablation=True,
                  test_used_for_fitting_or_selection=False,
                  protocol_sha256=sha256(Path(run) / 'protocol.json'))
    save_new(output, result)
    return result
