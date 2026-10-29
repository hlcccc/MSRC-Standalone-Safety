"""Portable, strict artifact IO and request-level split checks."""
import hashlib
import json
import math
from pathlib import Path
import re


ROLES = ('fit', 'calibration', 'test')


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            value.update(block)
    return value.hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key: ' + key)
        result[key] = value
    return result


def loads(text):
    def invalid(value):
        raise ValueError('nonfinite JSON number: ' + value)
    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            invalid(value)
        return result
    return json.loads(text, parse_constant=invalid, parse_float=finite_float, object_pairs_hook=_pairs)


def read_json(path):
    return loads(Path(path).read_text(encoding='utf-8'))


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [loads(line) for line in stream if line.strip()]


def save_new(path, value):
    """Never overwrite an earlier protocol or request record."""
    text = json.dumps(value, indent=2, allow_nan=False, ensure_ascii=True) + '\n'
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)


def beneath(root, name):
    root = Path(root).resolve(strict=True)
    relative = Path(name)
    if relative.is_absolute():
        raise ValueError('asset paths must be relative to their supplied root')
    path = (root / relative).resolve(strict=True)
    if not path.is_file() or not path.is_relative_to(root):
        raise ValueError('asset escaped its root or is not a file')
    return path


def record_path(root, role, rid):
    if role not in ('calibration', 'test'):
        raise ValueError('invalid record role')
    key = hashlib.sha256(rid.encode('utf-8')).hexdigest()
    root = Path(root).resolve()
    path = (root / role / (key + '.json')).resolve()
    if not path.is_relative_to(root):
        raise ValueError('record path escaped run root')
    return path


def load_inputs(path):
    rows, groups, exact = {}, {}, {}
    roles = {name: set() for name in ROLES}
    for row in read_jsonl(path):
        if not isinstance(row, dict):
            raise ValueError('request objects required')
        for field in ('id', 'group', 'role', 'question', 'image', 'image_sha256'):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError('nonempty request field required: ' + field)
        role, rid, group = row['role'], row['id'], row['group']
        if role not in ROLES or rid in rows:
            raise ValueError('unique IDs and fit/calibration/test roles required')
        if type(row.get('label')) is not int or row['label'] not in (0, 1):
            raise ValueError('binary request-risk labels required')
        if not re.fullmatch('[0-9a-f]{64}', row['image_sha256']):
            raise ValueError('lowercase SHA256 image digest required')
        if group in groups and groups[group] != role:
            raise ValueError('original-question group crossed roles')
        groups[group] = role
        fingerprint = (row['question'], row['image_sha256'])
        if fingerprint in exact and exact[fingerprint] != role:
            raise ValueError('exact image/question request crossed roles')
        exact[fingerprint] = role
        rows[rid] = row
        roles[role].add(rid)
    if any(not ids for ids in roles.values()):
        raise ValueError('manifest must include all three nonempty roles')
    return rows, roles


def source_hashes():
    return {path.name: sha256(path) for path in sorted(Path(__file__).parent.glob('*.py'))}


def fitted_model(path):
    import math
    import numpy as np
    from .signals import project

    model = read_json(path)
    if model.get('fitting_role') != 'fit':
        raise ValueError('fit-only direction provenance required')
    hold = model.get('hold_threshold')
    if type(hold) not in (float, int) or not math.isfinite(hold) or hold < 0:
        raise ValueError('finite fit-only hold threshold required')
    weights = np.asarray(model['weight'])
    if weights.ndim != 2 or weights.shape[0] != 3 or weights.shape[1] < 1:
        raise ValueError('three nonempty risk directions required')
    project(np.zeros(weights.shape), model)
    return model


def context(inputs, directions, generator_spec, max_tokens):
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError('positive token limit required')
    model = fitted_model(directions)
    identity = dict(inputs_sha256=sha256(inputs), directions_sha256=sha256(directions),
                    generator_spec_sha256=sha256(generator_spec), max_tokens=max_tokens,
                    code_sha256=source_hashes())
    if model.get('partition_sha256') != identity['inputs_sha256']:
        raise ValueError('directions must be fitted or explicitly bound to this partition')
    if model.get('generator_spec_sha256') != identity['generator_spec_sha256']:
        raise ValueError('directions generator identity differs')
    return identity
