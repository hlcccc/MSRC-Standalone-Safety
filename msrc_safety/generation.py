"""Local Qwen2.5-VL greedy decoding with causal post-token layer capture.

This adapter is repackaged from the historical experiment. It uses Transformers
private generation helpers, so the pinned runtime and native-greedy smoke check
are required. Importing the CPU policy does not import PyTorch or Transformers.
"""
import math
from pathlib import Path
import time

from .assets import beneath, read_json, sha256
from .policy import StandaloneController
from .signals import project, selected_layers


class LayerCapture:
    def __init__(self, decoder):
        self.decoder = decoder
        self.layers = selected_layers(len(decoder.layers))
        self.values, self.handles = {}, []

    def hook(self, number):
        def collect(module, args, result):
            value = result[0] if isinstance(result, (tuple, list)) else result
            if value.ndim != 3 or value.shape[0] != 1:
                raise ValueError('batch-one decoder activations required')
            self.values[number] = value[0, -1].detach().clone()
        return collect

    def __enter__(self):
        try:
            for number in self.layers:
                handle = self.decoder.layers[number - 1].register_forward_hook(self.hook(number))
                self.handles.append(handle)
        except Exception:
            self.__exit__()
            raise
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def result(self, torch):
        if set(self.values) != set(self.layers):
            raise ValueError('missing layer activations')
        return torch.stack([self.values[n] for n in self.layers]).float().cpu().numpy()


class Decoder:
    def __init__(self, model_dir, image_root, generator_spec, directions,
                 device, memory_limit_gib=20):
        import torch
        import transformers
        from transformers import AutoProcessor, AutoModelForImageTextToText

        if transformers.__version__ != '5.13.0':
            raise RuntimeError('this adapter requires transformers==5.13.0 and a native-greedy smoke check')
        if not torch.cuda.is_available() or not device.startswith('cuda:'):
            raise ValueError('an explicitly selected CUDA device is required for generation')
        if not math.isfinite(memory_limit_gib) or memory_limit_gib <= 0:
            raise ValueError('positive GPU allocation limit required')
        self.device = torch.device(device)
        total = torch.cuda.get_device_properties(self.device).total_memory
        if memory_limit_gib * 1024**3 > total:
            raise ValueError('allocation limit exceeds selected GPU capacity')
        torch.cuda.set_per_process_memory_fraction(memory_limit_gib * 1024**3 / total, self.device)
        torch.set_num_threads(2)
        torch.manual_seed(20261024)
        spec = read_json(generator_spec)['generator']
        if not spec.get('files'):
            raise ValueError('generator spec must pin all local model/processor files')
        for name, digest in spec['files'].items():
            if sha256(beneath(model_dir, name)) != digest:
                raise ValueError('generator file identity differs: ' + name)
        folder = str(Path(model_dir).resolve(strict=True))
        self.processor = AutoProcessor.from_pretrained(folder, local_files_only=True, max_pixels=1003520)
        self.model = AutoModelForImageTextToText.from_pretrained(
            folder, dtype=torch.bfloat16, device_map=device, local_files_only=True,
            attn_implementation='sdpa').eval()
        self.model.requires_grad_(False)
        candidates = [getattr(getattr(self.model, 'model', None), 'language_model', None),
                      getattr(self.model, 'model', None)]
        matches = [value for value in candidates if hasattr(value, 'layers') and hasattr(value, 'norm')]
        if len(matches) != 1:
            raise ValueError('expected one supported text decoder')
        self.decoder, self.torch = matches[0], torch
        self.image_root, self.directions = image_root, directions

    def inputs(self, row):
        from PIL import Image

        path = beneath(self.image_root, row['image'])
        if sha256(path) != row['image_sha256']:
            raise ValueError('source image bytes changed')
        text = self.processor.apply_chat_template([
            dict(role='user', content=[dict(type='image'), dict(type='text', text=row['question'])])
        ], tokenize=False, add_generation_prompt=True)
        with Image.open(path) as image:
            return self.processor(text=[text], images=[image.convert('RGB')],
                                  return_tensors='pt').to(self.device)

    def decode(self, row, thresholds, max_tokens=128, full_shadow=False):
        torch, model = self.torch, self.model
        controls = {name: StandaloneController(value, self.directions['hold_threshold'])
                    for name, value in thresholds.items()}
        decisions, events = {}, {name: [] for name in controls}
        tokens, trace = [], []
        started = time.time_ns()
        stop_reason = 'max_tokens'
        with torch.inference_mode(), LayerCapture(self.decoder) as capture:
            inputs = self.inputs(row)
            ids = inputs['input_ids']
            kwargs = {key: value for key, value in inputs.items() if key != 'input_ids'}
            kwargs.update(use_cache=True, logits_to_keep=1)
            kwargs['position_ids'] = model._prepare_position_ids_for_generation(ids, dict(kwargs, input_ids=ids))
            config, _ = model._prepare_generation_config(None, do_sample=False, max_new_tokens=max_tokens)
            config.max_length = ids.shape[1] + max_tokens
            model._prepare_special_tokens(config, kwargs_has_attention_mask=True, device=ids.device)
            processors = model._get_logits_processor(
                config, input_ids_seq_length=ids.shape[1], device=ids.device, model_kwargs=kwargs)
            out = model(**model.prepare_inputs_for_generation(ids, is_first_iteration=True, **kwargs),
                        output_hidden_states=False, return_dict=True)
            eos = model.generation_config.eos_token_id
            eos = set(eos if isinstance(eos, list) else [eos])
            for position in range(1, max_tokens + 1):
                token = processors(ids, out.logits[:, -1].float()).argmax(-1, keepdim=True)
                tid = int(token.item())
                if tid in eos:
                    stop_reason = 'eos'
                    break
                kwargs = model._update_model_kwargs_for_generation(out, kwargs)
                ids = torch.cat([ids, token], -1)
                prepared = model.prepare_inputs_for_generation(
                    ids, next_sequence_length=1, is_first_iteration=False, **kwargs)
                if prepared['input_ids'].shape[1] != 1:
                    raise ValueError('cached decode must consume exactly one generated token')
                capture.values.clear()
                out = model(**prepared, output_hidden_states=False, return_dict=True)
                scores = project(capture.result(torch), self.directions).tolist()
                tokens.append(tid)
                trace.append(scores)
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
            if not tokens:
                raise ValueError('empty generation is an error, not an inferred safe answer')
            for name, controller in controls.items():
                if name not in decisions:
                    decisions[name] = controller.finish()
        arms = {name: dict(decision=value, events=events[name]) for name, value in decisions.items()}
        return dict(layer_score_trace=trace, token_ids=tokens, arms=arms,
                    start_ns=started, end_ns=time.time_ns(), stop_reason=stop_reason,
                    full_shadow=full_shadow, plugguard_used=False)

    def smoke(self, calibration_rows):
        checks = []
        suites = sorted({row.get('suite', 'unspecified') for row in calibration_rows})
        for suite in suites:
            row = next(value for value in calibration_rows if value.get('suite', 'unspecified') == suite)
            result = self.decode(row, {}, max_tokens=32, full_shadow=True)
            with self.torch.inference_mode():
                inputs = self.inputs(row)
                native = self.model.generate(**inputs, max_new_tokens=32, do_sample=False)
            eos = self.model.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            reference = [value for value in native[0, inputs['input_ids'].shape[1]:].tolist() if value not in eos]
            if reference != result['token_ids']:
                raise ValueError('decoding differs from native greedy: ' + row['id'])
            checks.append(dict(suite=suite, exact_greedy_match=True, tokens=len(reference)))
        return dict(calibration_only=True, no_online_guard_model=True, checks=checks)
