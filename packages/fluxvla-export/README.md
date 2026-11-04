# fluxvla-export

Minimal FluxVLA build frontend consumed by FluxHermes. FluxVLA owns model
construction, export boundaries and model-specific execution; FluxHermes owns
ONNX export, TensorRT building, parity checks, benchmarks and bundles.

## Current scope

Implemented: versioned deployment configuration, safe artifact readers, tensor
contracts, and validated API dispatch. **No model recipe is registered yet.**
There is no PI0.5 model construction, checkpoint conversion, ONNX/TensorRT
export or model parity result in this package yet.

Schema 1 describes the proposed `pi05-v1` family, not a particular robot. It
accepts static six-input fixtures, complete native dense weights, plain Euler
inference and FP32 master parameters with either FP32 or BF16-autocast reference
execution. FP16 execution, per-parameter storage dtype maps, dynamic fixture
profiles, unmerged adapters, RTC and external checkpoint naming are explicitly
unsupported in this schema. This is a configuration boundary, not a claim of
validated model execution.

## Install and test

```bash
# From the FluxVLA repository root; does not install the root FluxVLA package.
python -m pip install './packages/fluxvla-export[test]'
python -m unittest discover -s packages/fluxvla-export/tests -v
```

The readers use PyTorch 2.8 and safetensors 0.6.2-compatible releases. The runtime
also declares `packaging`: safetensors' torch integration imports it without a
core dependency declaration. The test extra supplies NumPy, required by
safetensors' test-file writer. Model libraries
such as Transformers are not needed by this block. Importing the public package
or parsing a config does not import torch, safetensors or the training stack.
Missing test dependencies are errors, not reasons to silently skip tensor tests.

## Explicit deployment configuration

```python
from fluxvla_export import load_config, prepare_tensorrt

config = load_config('export_config.json', recipe='pi05-v1')

# Requires a future real recipe; currently raises RecipeNotFoundError after
# validating the configuration. It does not synthesize a model or outputs.
prepared = prepare_tensorrt(
    recipe='pi05-v1',
    config='export_config.json',
    checkpoint='model.safetensors',
    inputs='export_inputs.safetensors',
    device='cuda:0',
)
```

`config` is a required argument, including on `ExportRecipe.prepare`. The public
entry takes a UTF-8 JSON path; the recipe receives the parsed, frozen
`DeploymentConfig`. Python training configs are never executed and adjacent
files are never discovered automatically. Only `cpu`, `cuda` and `cuda:<index>`
are accepted as device syntax; hardware availability belongs to real model
preparation, not this preflight.

### Schema 1 fields

Every listed field is required and every object rejects unknown fields. JSON
object key order is irrelevant; input tensors are returned in the canonical
order below. No implicit architecture defaults are filled in.

| Object | Required fields / values |
| --- | --- |
| Root | `schema_version: 1`, `recipe: "pi05-v1"`, `checkpoint`, `inputs`, `model`, `inference`, `precision`, `preprocessing` |
| `checkpoint` | `format: "fluxvla-native"`, `kind: "dense"`, `sha256` |
| `inputs` | `format: "safetensors"`, `sha256`, `tensors` |
| Each entry in `inputs.tensors` | `dtype`, `shape` (positive integer array, no dynamic dimensions) |
| `model` | `type: "pi05"`, `vision`, `backbone`, `expert`, `projector`, `action_in_proj`, `action_out_proj`, `time_mlp_in`, `time_mlp_out`, `proj_width`, `state_dim`, `action_dim`, `action_horizon`, `state_projection: false` |
| `model.vision` | `type: "siglip"`, `hidden_size`, `intermediate_size`, `num_hidden_layers`, `num_attention_heads`, `num_channels: 3`, `image_size`, `patch_size`, `hidden_activation: "gelu_pytorch_tanh"`, `layer_norm_eps`, `attention_dropout: 0`, `vision_use_head: false` |
| `model.backbone` and `model.expert` | `type: "condition_gemma"`, `hidden_size`, `intermediate_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `head_dim`, `vocab_size`, `max_position_embeddings`, `hidden_activation: "gelu_pytorch_tanh"`, `rms_norm_eps`, `rope_theta`, `attention_bias: false`, `attention_dropout: 0`, `use_adarms`, `adarms_cond_dim`, `use_cache: true`, `pad_token_id`, `bos_token_id`, `eos_token_id` |
| Each projector | `type: "linear"`, `in_dim`, `out_dim`, `bias: true` |
| `inference` | `mode: "plain"`, `solver: "euler"`, `num_steps`, `rtc: false` |
| `precision` | `parameter_dtype: "float32"`, `autocast_dtype: null` or `"bfloat16"`, `attention_backend: "sdpa"` or `"eager"`, booleans `flow_fp32`, `vision_stem_fp32`, `residual_fp32`, `vision_residual_fp32`, `adarms_fp32`, `attention_math_fp32`, and integer arrays `backbone_fp32_layers`, `expert_fp32_layers` |
| `preprocessing` | `id`, `tokenizer_id`, `tokenizer_sha256`, `prompt_id`, `normalization_id`, `normalization_sha256`, `state_semantics`, `action_semantics`, `image_transform_id`, `view_order`, `state_in_prompt` |

Shapes, widths, cache geometry and projection dimensions must agree. Head
widths are not inferred from hidden width: the ordinary PI0.5 expert uses width
1024 with eight 256-wide query heads. Backbone conditioning must be null;
expert AdaRMS must be enabled with conditioning width equal to `proj_width`.
All projection dimensions are checked. FP32 attention math requires the `sdpa`
backend. Protected layer indices must be unique and in range.

`state_dim` and `action_dim` describe padded model tensor dimensions, not the
robot's native command dimensions. `action_horizon` is the model window, not the
Euler step count or how many commands the robot executes. All SHA-256 fields
are lowercase 64-character hex strings; artifact digests bind the exact file
bytes. Duplicate JSON keys, non-finite numbers, unknown schema versions and
implicit bool/string-to-number conversions are rejected.

Preprocessing identifiers and digests are declarations from the training
producer. They are not proof that images were ordered or normalized correctly;
the reader does not have tokenizer/statistics assets and cannot independently
verify those transformations. Do not invent these values for real deployment.
A training-config-to-JSON exporter is not implemented in this block.

### Fixture boundary

```text
images       [B, 3*V, H, W]  float32 or bfloat16
img_masks    [B, V]          bool, True means valid
lang_tokens  [B, L]          int64, including padded IDs in vocabulary
lang_masks   [B, L]          bool, True means valid
states       [B, S]          float32 or bfloat16
noise        [B, T, A]       float32, explicit starting noise
```

BF16 image/state inputs require BF16 autocast in this schema. View count and
order come from `preprocessing.view_order`. Masks retain their complete pattern;
they are not reduced to lengths. Training supervision `action_masks` is not a
normal policy input. For state-in-prompt configurations, state tokens must
already be produced using the matching training preprocessing.

## Artifact readers

```python
from fluxvla_export import load_checkpoint, load_config, load_fixture

config = load_config('export_config.json', recipe='pi05-v1')
inputs = load_fixture(
    'export_inputs.safetensors',
    expected_sha256=config.inputs_sha256,
    specs=config.input_specs,
)
# model must be a real, correctly constructed module; construction is not yet
# provided by fluxvla-export. A complete expected state is always required.
state = load_checkpoint(
    'model.safetensors',
    expected_sha256=config.checkpoint_sha256,
    expected_state=model.state_dict(),
)
model.load_state_dict(state, strict=True)
```

- File readers currently target POSIX hosts (Linux/macOS). Nonblocking opens
  reject non-regular files such as FIFOs rather than hanging before validation.
- Both readers return actual CPU torch tensors. They reject changed bytes,
  malformed/truncated headers, duplicate keys, overlapping/gapped data, extra
  trailing bytes, missing/extra tensor names, wrong shapes/dtypes and NaN/Inf.
- Checkpoint comparison is against the caller's complete expected state,
  including scalar/empty buffers. There is no prefix guessing, key mapping,
  silent dtype conversion, pickle fallback or mutation of that expected state.
  This proves compatibility with the supplied state, not independently with
  a PI0.5 architecture. Native model key verification needs the future recipe.
- The generic fixture reader follows `TensorSpec` ordering and supports its
  min/max profiles; the current deployment JSON intentionally requires static
  shapes. Vocabulary checks are PI0.5 API preflight, not a generic fixture rule.
- A reader verifies and decodes the same immutable byte snapshot, so replacing
  a path cannot substitute different bytes after verification. Loading retains
  file bytes plus decoded tensors in CPU memory; large-model streaming loading
  is not implemented. Header-only preflight streams the hash without creating
  weight tensors. A path returned by preflight must be revalidated when loaded.
- `ArtifactError`, `ConfigError` and `ContractError` report the artifact/field or
  contract that failed. Untrusted safetensors metadata cannot relax validation.

API dispatch tests temporarily register a test-only Identity harness and use
real safetensors files; that is a dispatch test, not a model or GPU parity test.
The independent CI builds a wheel, installs it without the FluxVLA source tree,
and runs copied tests against the installed package on Python 3.10 and 3.12.
