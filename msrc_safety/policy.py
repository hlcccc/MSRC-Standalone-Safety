"""Independent three-layer warning policy: no PlugGuard scores or gate."""
import math


class Evidence:
    """The frozen multilayer experiment's causal equation, without its base gate."""

    def __init__(self):
        self.mean = self.value = 0.

    def step(self, values):
        if len(values) != 3 or not all(math.isfinite(float(x)) for x in values):
            raise ValueError('three finite layer scores required')
        values = [float(x) for x in values]
        score = sum(values) / 3
        disagreement = math.sqrt(sum((x - score) ** 2 for x in values) / 3)
        value = .9 * self.value + max(0., score - .5) + .25 * max(0., score - self.mean)
        mean = .8 * self.mean + .2 * score
        if not all(math.isfinite(x) for x in (value, mean, disagreement)):
            raise ValueError('nonfinite evidence')
        self.mean, self.value = mean, value
        return dict(semantic=max(0., score), temporal=value, disagreement=disagreement)


class StandaloneController:
    def __init__(self, threshold, hold_threshold, base_buffer=16, hold_buffer=64):
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or
               not math.isfinite(x) or x < 0 for x in (threshold, hold_threshold)):
            raise ValueError('finite nonnegative thresholds required')
        if (type(base_buffer) is not int or type(hold_buffer) is not int or
                base_buffer < 16 or hold_buffer < base_buffer):
            raise ValueError('invalid buffer lengths')
        self.threshold, self.hold_threshold = threshold, hold_threshold
        self.base_buffer, self.hold_buffer = base_buffer, hold_buffer
        self.evidence = Evidence()
        self.generated = self.released = self.pending = 0
        self.state, self.hold = 'NORMAL', False
        self.scores = dict(semantic=0., temporal=0., disagreement=0.)

    def decision(self, alarm=False, reason='continue'):
        return dict(alarm=alarm, reason=reason, state=self.state, generated=self.generated,
                    released=self.released, pending=self.pending,
                    buffer_limit=self.hold_buffer if self.state == 'HOLD' else self.base_buffer,
                    **self.scores)

    def step(self, values):
        if self.state in ('BLOCKED', 'FINISHED'):
            raise RuntimeError('controller is terminal')
        scores = self.evidence.step(values)
        self.scores = scores
        self.generated += 1
        self.pending += 1
        if scores['temporal'] > self.threshold:
            self.state, self.pending = 'BLOCKED', 0
            return self.decision(True, 'standalone_multilayer_temporal_alarm')
        if scores['disagreement'] > self.hold_threshold:
            self.hold = True
        elif scores['disagreement'] <= .5 * self.hold_threshold:
            self.hold = False
        self.state = 'HOLD' if self.hold else 'NORMAL'
        limit = self.hold_buffer if self.hold else self.base_buffer
        count = max(0, self.pending - limit)
        self.pending -= count
        self.released += count
        return self.decision()

    def finish(self):
        if self.state in ('BLOCKED', 'FINISHED'):
            raise RuntimeError('controller is terminal')
        self.released += self.pending
        self.pending = 0
        self.state = 'FINISHED'
        return self.decision(False, 'finished_without_alarm')


def metrics(rows):
    counts = dict(TP=0, TN=0, FP=0, FN=0)
    for row in rows:
        if type(row.get('label')) is not int or row['label'] not in (0, 1) or type(row.get('warn')) is not bool:
            raise ValueError('binary request label and warning required')
        counts[('TP' if row['warn'] else 'FN') if row['label'] else ('FP' if row['warn'] else 'TN')] += 1
    n = sum(counts.values())
    ratio = lambda a, b: a / b if b else None
    recall = ratio(counts['TP'], counts['TP'] + counts['FN'])
    specificity = ratio(counts['TN'], counts['TN'] + counts['FP'])
    return dict(n=n, **counts, accuracy=ratio(counts['TP'] + counts['TN'], n),
                recall=recall, benign_false_positive_rate=None if specificity is None else 1 - specificity,
                precision=ratio(counts['TP'], counts['TP'] + counts['FP']),
                balanced_accuracy=None if recall is None or specificity is None else (recall + specificity) / 2)


def select_request_threshold(rows):
    """Maximize calibration request-label accuracy; strict >, lowest-threshold tie."""
    seen, groups = set(), {0: set(), 1: set()}
    for row in rows:
        if row.get('role') != 'calibration':
            raise ValueError('calibration role required; test cannot select a threshold')
        if (not isinstance(row.get('id'), str) or not row['id'] or row['id'] in seen or
                not isinstance(row.get('group'), str) or not row['group'] or
                type(row.get('label')) is not int or row['label'] not in (0, 1)):
            raise ValueError('unique calibration identities and valid request labels required')
        if not math.isfinite(row['score']) or row['score'] < 0:
            raise ValueError('nonnegative finite request maximum required')
        seen.add(row['id'])
        groups[row['label']].add(row['group'])
    if any(len(v) < 20 for v in groups.values()):
        raise ValueError('at least twenty independent groups per class required')
    candidates = sorted({0., *(float(r['score']) for r in rows)})
    correct = lambda t: sum((r['score'] > t) == bool(r['label']) for r in rows)
    threshold = max(candidates, key=lambda t: (correct(t), -t))
    scores = metrics([dict(label=r['label'], warn=r['score'] > threshold) for r in rows])
    return dict(threshold=threshold, objective='maximum_calibration_request_label_accuracy',
                comparison='strict_greater_than', tie_break='lowest_nonnegative_threshold',
                calibration=scores, candidates=len(candidates), test_used=False)
