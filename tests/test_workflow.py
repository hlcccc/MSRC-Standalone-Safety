import copy
import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

from msrc_safety import workflow
from msrc_safety.assets import beneath, load_inputs, loads, read_json, record_path, save_new
from msrc_safety.policy import StandaloneController


class FakeDecoder:
    """Deterministic score fixtures only; this is not a model or GPU experiment."""
    constructions = 0

    def __init__(self, model_dir, image_root, generator_spec, directions, device, memory_limit_gib):
        self.directions = directions
        FakeDecoder.constructions += 1

    def smoke(self, rows):
        return dict(calibration_only=True, no_online_guard_model=True,
                    checks=[dict(suite=name, exact_greedy_match=True, tokens=32)
                            for name in sorted({row['suite'] for row in rows})])

    def decode(self, row, thresholds, max_tokens=128, full_shadow=False):
        controls = {name: StandaloneController(value, self.directions['hold_threshold'])
                    for name, value in thresholds.items()}
        decisions, events = {}, {name: [] for name in controls}
        trace, tokens = [], []
        started = time.time_ns()
        stop_reason = 'max_tokens'
        for position in range(1, max_tokens + 1):
            scores = [1. if row['label'] else -1.] * 3
            trace.append(scores)
            tokens.append(position)
            for name, controller in controls.items():
                if name in decisions:
                    continue
                decision = controller.step(scores)
                events[name].append(dict(position=position, score_ns=time.time_ns(), decision=decision))
                if decision['alarm']:
                    decisions[name] = decision
            if controls and not full_shadow and len(decisions) == len(controls):
                stop_reason = 'all_arms_blocked'
                break
        for name, controller in controls.items():
            if name not in decisions:
                decisions[name] = controller.finish()
        return dict(layer_score_trace=trace, token_ids=tokens, start_ns=started,
                    end_ns=time.time_ns(), stop_reason=stop_reason, full_shadow=full_shadow,
                    plugguard_used=False,
                    arms={name: dict(decision=value, events=events[name]) for name, value in decisions.items()})


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.inputs = self.root / 'inputs.jsonl'
        self.spec = self.root / 'generator.spec.json'
        self.directions = self.root / 'directions.json'
        self.run = self.root / 'run'
        self.rows = []
        for role, count in (('fit', 64), ('calibration', 40), ('test', 4)):
            for index in range(count):
                rid = role + '-' + str(index)
                self.rows.append(dict(id=rid, group=rid, role=role, label=index % 2,
                                      question='synthetic-' + rid, image='placeholder.png',
                                      image_sha256='0' * 64, suite='synthetic'))
        self.write_inputs(self.rows)
        save_new(self.spec, dict(generator=dict(repo='synthetic/no-model', revision='fixture', files={})))
        records = [dict(id=row['id'], group=row['group'], role='fit', label=row['label'],
                        position=1, label_source='response_prefix_judge')
                   for row in self.rows if row['role'] == 'fit']
        self.checkpoints, self.features = self.root / 'checkpoints.jsonl', self.root / 'features.npz'
        self.checkpoints.write_text(''.join(json.dumps(row) + '\n' for row in records), encoding='utf-8')
        np.savez(self.features, layers=np.random.default_rng(19).normal(size=(64, 3, 8)))
        workflow.fit(self.inputs, self.checkpoints, self.features, self.spec, self.directions)

    def write_inputs(self, rows):
        self.inputs.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')

    def collect(self):
        return workflow.collect(self.inputs, self.directions, self.spec, self.run, self.root,
                                self.root, 'cuda:0', max_tokens=32, decoder_factory=FakeDecoder)

    def freeze(self):
        self.collect()
        return workflow.freeze(self.inputs, self.directions, self.spec, self.run, inherited_threshold=6.)

    def run_test(self):
        return workflow.test(self.inputs, self.directions, self.spec, self.run, self.root,
                             self.root, 'cuda:0', decoder_factory=FakeDecoder)

    def test_complete_fixture_workflow_and_idempotent_resume(self):
        self.assertEqual(self.collect()['completed'], 40)
        constructions = FakeDecoder.constructions
        self.assertTrue(self.collect()['resumed'])
        self.assertEqual(FakeDecoder.constructions, constructions)
        selection = workflow.freeze(self.inputs, self.directions, self.spec, self.run, inherited_threshold=6.)
        self.assertEqual(selection['thresholds']['standalone_recalibrated'], 0.)
        self.assertEqual(self.run_test()['completed'], 4)
        constructions = FakeDecoder.constructions
        self.assertTrue(self.run_test()['resumed'])
        self.assertEqual(FakeDecoder.constructions, constructions)
        result = workflow.report(self.inputs, self.directions, self.spec, self.run, self.root / 'report.json')
        self.assertEqual(result['status'], 'COMPLETE')
        self.assertEqual(result['causal_replay_verified_records'], 4)
        self.assertEqual(result['primary']['standalone_recalibrated']['accuracy'], 1.)
        self.assertEqual(result['primary']['standalone_inherited']['accuracy'], .5)

    def test_test_cannot_run_before_freeze(self):
        with self.assertRaises(FileNotFoundError):
            self.run_test()

    def test_freeze_requires_all_calibration(self):
        self.collect()
        record_path(self.run, 'calibration', 'calibration-0').unlink()
        with self.assertRaisesRegex(ValueError, 'all calibration'):
            workflow.freeze(self.inputs, self.directions, self.spec, self.run)

    def test_recalibration_after_freeze_is_rejected(self):
        self.freeze()
        with self.assertRaises(ValueError):
            self.collect()
        with self.assertRaises(ValueError):
            workflow.freeze(self.inputs, self.directions, self.spec, self.run)

    def test_test_coverage_is_not_inferred_from_calibration(self):
        self.freeze()
        result = workflow.report(self.inputs, self.directions, self.spec, self.run, self.root / 'partial.json')
        self.assertEqual(result['status'], 'PARTIAL_NOT_FINAL_ACCURACY')
        self.assertEqual(result['completed'], 0)
        self.assertIsNone(result['primary']['standalone_recalibrated']['accuracy'])

    def test_changed_calibration_bytes_fail_before_test(self):
        self.freeze()
        path = record_path(self.run, 'calibration', 'calibration-0')
        path.write_text(path.read_text(encoding='utf-8') + ' ', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'calibration coverage or bytes'):
            self.run_test()

    def test_causal_event_tampering_fails(self):
        self.freeze()
        self.run_test()
        path = record_path(self.run, 'test', 'test-1')
        record = read_json(path)
        record['arms']['standalone_recalibrated']['events'][0]['decision']['released'] = 999
        path.write_text(json.dumps(record), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'causal decision replay'):
            workflow.report(self.inputs, self.directions, self.spec, self.run, self.root / 'bad.json')

    def test_partition_and_generator_identity_are_bound(self):
        self.freeze()
        self.spec.write_text('{}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'generator identity'):
            self.run_test()

    def test_original_groups_cannot_cross_roles(self):
        rows = copy.deepcopy(self.rows)
        rows[-1]['group'] = rows[0]['group']
        self.write_inputs(rows)
        with self.assertRaisesRegex(ValueError, 'group crossed roles'):
            load_inputs(self.inputs)

    def test_exact_requests_cannot_cross_roles(self):
        rows = copy.deepcopy(self.rows)
        rows[-1]['question'] = rows[0]['question']
        self.write_inputs(rows)
        with self.assertRaisesRegex(ValueError, 'exact image/question'):
            load_inputs(self.inputs)

    def test_import_is_explicit_and_drops_legacy_policy_fields(self):
        legacy = read_json(self.directions)
        legacy['temporal'] = 123.
        legacy['unused_legacy_gate'] = 999.
        path, bound = self.root / 'old.json', self.root / 'bound.json'
        save_new(path, legacy)
        workflow.bind_directions(self.inputs, path, self.spec, bound)
        result = read_json(bound)
        self.assertEqual(result['weight'], legacy['weight'])
        self.assertTrue(result['source_fit_audit_required'])
        self.assertNotIn('unused_legacy_gate', result)
        self.assertNotIn('temporal', result)

    def test_outputs_never_overwrite_existing_files(self):
        with self.assertRaises(FileExistsError):
            save_new(self.spec, {})

    def test_strict_json_and_asset_boundary(self):
        for value in ('{"x": NaN}', '{"x": 1e999}', '{"x": 0, "x": 1}'):
            with self.assertRaises(ValueError):
                loads(value)
        assets = self.root / 'assets'
        assets.mkdir()
        with self.assertRaises(ValueError):
            beneath(assets, '../generator.spec.json')
