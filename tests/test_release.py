import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

from msrc_safety.cli import demo


ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_frozen_core_bytes(self):
        expected = {
            'policy.py': '88945e1f5c5f57b6deab34514ba22408e0fa9be6f164bfa7b189052782b58adc',
            'signals.py': 'dc6fe8fda8473fbc29f338a7295f6cf66b1802cd3ee1c9254a821d73685c36a5',
        }
        for name, digest in expected.items():
            self.assertEqual(hashlib.sha256((ROOT / 'msrc_safety' / name).read_bytes()).hexdigest(), digest)

    def test_no_old_project_or_guard_imports(self):
        forbidden = {'plugguard', 'safety_lab', 'replay', 'cloud_prepare', 'hlc_multilayer_run'}
        for path in (ROOT / 'msrc_safety').glob('*.py'):
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                if isinstance(node, ast.Import):
                    self.assertFalse({name.name.split('.')[0] for name in node.names} & forbidden)
                elif isinstance(node, ast.ImportFrom):
                    self.assertNotIn((node.module or '').split('.')[0], forbidden)

    def test_cpu_import_does_not_import_model_runtime(self):
        code = 'import sys, msrc_safety, msrc_safety.cli; assert "torch" not in sys.modules; assert "transformers" not in sys.modules'
        subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True, capture_output=True)

    def test_demo_is_explicitly_synthetic(self):
        result = demo()
        self.assertTrue(result['synthetic'])
        self.assertFalse(result['acceptance_result'])
        self.assertEqual(result['events'][-1]['decision']['state'], 'BLOCKED')

    def test_historical_metrics_match_counts_and_expose_limitations(self):
        from msrc_safety.policy import metrics
        report = json.loads((ROOT / 'results/historical_20261028/report.json').read_text(encoding='utf-8'))
        self.assertEqual(report['completed'], 3360)
        self.assertTrue(report['harmful_token_early_warning_accuracy_not_proven'])
        for arm in report['primary'].values():
            rows = []
            for key, label, warn in (('TP', 1, True), ('TN', 0, False), ('FP', 0, True), ('FN', 1, False)):
                rows.extend([dict(label=label, warn=warn)] * arm[key])
            self.assertEqual(metrics(rows), arm)
        provenance = json.loads((ROOT / 'results/historical_20261028/provenance.json').read_text(encoding='utf-8'))
        self.assertFalse(provenance['portable_gpu_rerun_performed_for_this_release'])
