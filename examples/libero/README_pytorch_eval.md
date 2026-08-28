# PyTorch LIBERO Evaluation Setup

This guide reproduces the local, headless PyTorch setup used to evaluate both the
LIBERO-fine-tuned pi0 checkpoint and the raw pi0 base checkpoint. It was tested on
Ubuntu 22.04 with an NVIDIA A100 40 GB GPU.

## 1. System packages

Install `uv` and the libraries needed for headless MuJoCo rendering:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
sudo apt-get update
sudo apt-get install -y libegl1 libegl-mesa0 libosmesa6 libosmesa6-dev
```

The evaluations below use OSMesa, which does not require an X server.

## 2. Initialize the repository

From the repository root:

```bash
git submodule update --init --recursive third_party/libero
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

The policy server uses the root Python 3.11 environment. LIBERO requires a
separate Python 3.8 client environment:

```bash
uv venv --python 3.8 examples/libero/.venv
uv pip sync \
  examples/libero/requirements.txt \
  third_party/libero/requirements.txt \
  --python examples/libero/.venv/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cu113 \
  --index-strategy=unsafe-best-match
uv pip install \
  --python examples/libero/.venv/bin/python \
  -e packages/openpi-client \
  -e third_party/libero
```

On its first run, LIBERO asks whether to use a custom dataset directory. Answer
`N` to create the default `~/.libero/config.yaml`.

## 3. Apply the required Transformers patch

The PyTorch implementation requires the repository's patched Transformers files:

```bash
cp -r src/openpi/models_pytorch/transformers_replace/* \
  .venv/lib/python3.11/site-packages/transformers/
```

This assumes the root environment uses Python 3.11. The repository pins
Transformers 4.53.2. See the warning in the root README: because `uv` may use
hardlinks, this copy can also affect the cached Transformers package.

Verify the GPU environment:

```bash
uv run python - <<'PY'
import torch
import transformers
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
PY
```

## 4. Convert the LIBERO-fine-tuned pi0 checkpoint

Download the JAX checkpoint:

```bash
uv run python - <<'PY'
from openpi.shared import download

print(download.maybe_download("gs://openpi-assets/checkpoints/pi0_libero"))
PY
```

Convert it to BF16 PyTorch weights:

```bash
uv run examples/convert_jax_model_to_pytorch.py \
  --checkpoint-dir ~/.cache/openpi/openpi-assets/checkpoints/pi0_libero \
  --config-name pi0_libero \
  --output-path ~/.cache/openpi/openpi-assets/checkpoints/pi0_libero_pytorch \
  --precision bfloat16
```

The conversion script in this revision does not copy checkpoint-local assets, so
copy the LIBERO normalization statistics explicitly:

```bash
cp -r \
  ~/.cache/openpi/openpi-assets/checkpoints/pi0_libero/assets \
  ~/.cache/openpi/openpi-assets/checkpoints/pi0_libero_pytorch/
```

## 5. Run one rollout

Start the policy server in one terminal:

```bash
TORCHDYNAMO_DISABLE=1 uv run scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config pi0_libero \
  --policy.dir ~/.cache/openpi/openpi-assets/checkpoints/pi0_libero_pytorch
```

`TORCHDYNAMO_DISABLE=1` avoids a several-minute `max-autotune` compilation on the
first request. Inference still uses PyTorch and CUDA in eager mode.

Run one task and one rollout in a second terminal:

```bash
PYTHONPATH="$PWD/third_party/libero" \
MUJOCO_GL=osmesa \
PYOPENGL_PLATFORM=osmesa \
examples/libero/.venv/bin/python examples/libero/main.py \
  --args.host 127.0.0.1 \
  --args.num-tasks 1 \
  --args.num-trials-per-task 1 \
  --args.video-out-path data/libero/videos/pi0_pytorch_smoke
```

## 6. Run parallel evaluation

One pi0 PyTorch server uses approximately 7.45 GiB during active inference. Four
servers used approximately 29.8 GiB on the A100 40 GB.

Start four servers on ports 8000 through 8003:

```bash
for port in 8000 8001 8002 8003; do
  TORCHDYNAMO_DISABLE=1 uv run scripts/serve_policy.py \
    --port "$port" \
    policy:checkpoint \
    --policy.config pi0_libero \
    --policy.dir ~/.cache/openpi/openpi-assets/checkpoints/pi0_libero_pytorch \
    >"/tmp/pi0_server_${port}.log" 2>&1 &
done
```

Partition 50 initial states across four clients:

```bash
counts=(13 13 12 12)
offsets=(0 13 26 38)

for worker in 0 1 2 3; do
  port=$((8000 + worker))
  PYTHONPATH="$PWD/third_party/libero" \
  MUJOCO_GL=osmesa \
  PYOPENGL_PLATFORM=osmesa \
  examples/libero/.venv/bin/python examples/libero/main.py \
    --args.host 127.0.0.1 \
    --args.port "$port" \
    --args.num-tasks 1 \
    --args.num-trials-per-task "${counts[$worker]}" \
    --args.initial-state-offset "${offsets[$worker]}" \
    --args.video-out-path "data/libero/videos/pi0_pytorch_50/worker_${worker}" \
    >"/tmp/pi0_client_${worker}.log" 2>&1 &
done

wait
```

The evaluator's `--args.initial-state-offset` option prevents workers from using
the same initial states. Episode indices are included in video filenames to avoid
overwrites.

## 7. Evaluate raw pi0 base weights

The raw base model has no LIBERO normalization statistics. To interface it with
LIBERO without changing its learned weights, convert `pi0_base` with the
`pi0_libero` architecture/configuration and attach only the LIBERO normalization
file:

```bash
uv run python - <<'PY'
from openpi.shared import download

print(download.maybe_download("gs://openpi-assets/checkpoints/pi0_base"))
PY

uv run examples/convert_jax_model_to_pytorch.py \
  --checkpoint-dir ~/.cache/openpi/openpi-assets/checkpoints/pi0_base \
  --config-name pi0_libero \
  --output-path ~/.cache/openpi/openpi-assets/checkpoints/pi0_base_pytorch_libero_eval \
  --precision bfloat16

mkdir -p \
  ~/.cache/openpi/openpi-assets/checkpoints/pi0_base_pytorch_libero_eval/assets/physical-intelligence/libero
cp \
  ~/.cache/openpi/openpi-assets/checkpoints/pi0_libero/assets/physical-intelligence/libero/norm_stats.json \
  ~/.cache/openpi/openpi-assets/checkpoints/pi0_base_pytorch_libero_eval/assets/physical-intelligence/libero/
```

Use `pi0_base_pytorch_libero_eval` as `--policy.dir` in the server commands above.
Only the input/output transforms and action normalization are LIBERO-specific;
the model parameters remain the raw base pretrained weights.

## Observed smoke-test results

- LIBERO-fine-tuned pi0: 49/50 successes on the first `libero_spatial` task.
- Raw pi0 base weights: 0/20 successes on the same task.

These numbers are smoke-test observations, not a complete LIBERO benchmark run.
