# Profiling PyTorch on Modal 🔥

Quick benchmarks that use the [PyTorch profiler](https://docs.pytorch.org/docs/stable/profiler.html) to see what actually runs on the GPU — kernel launches, fusion, and the wins from `torch.compile`, SDPA, and pre-built kernels from the [kernels hub](https://huggingface.co/blog/hello-hf-kernels). Each script runs on an A100 via [Modal](https://modal.com), no local GPU needed, and writes its trace and summary table to `traces/<script_name>/`.

These benchmarks were derived from Hugging Face's [Profiling in PyTorch](https://huggingface.co/blog/torch-profiler) series, which is well worth reading.

## Setup

```bash
curl -LsSf uvx.sh/modal/install.sh | sh
modal setup
```

## Running a benchmark

```bash
modal run 01_matmul_add.py
```

Every script exposes its knobs as flags on the Modal entrypoint:

```bash
modal run 01_matmul_add.py --size 4096 --dtype fp32 --no-compile
modal run 04_c_sdpa_attention.py --backend flash --seq 4096
```

## The benchmarks

| Script | What it shows |
| --- | --- |
| [01_matmul_add.py](01_matmul_add.py) | `matmul` + `add` baseline; `--compile` shows compile time in a cold trace. |
| [02_linear.py](02_linear.py) | One `nn.Linear` forward — a matmul and a bias add as kernels. |
| [03_simple_mlp.py](03_simple_mlp.py) | A GeGLU MLP in plain ops — many separate kernels in eager mode. |
| [03_kernels_mlp.py](03_kernels_mlp.py) | The same MLP fused via `LigerGEGLUMLP` from the [`kernels`](https://github.com/huggingface/kernels) hub. |
| [04_a_naive_attention.py](04_a_naive_attention.py) | Causal attention spelled out, materializing the full `[seq, seq]` scores. |
| [04_b_inplace_ops_attention.py](04_b_inplace_ops_attention.py) | Same attention with in-place ops — memory traffic vs 04_a. |
| [04_c_sdpa_attention.py](04_c_sdpa_attention.py) | `F.scaled_dot_product_attention`, no mask materialized; `--backend` picks the kernel. |
| [04_d_kernels_attention.py](04_d_kernels_attention.py) | FlashAttention from the hub via `kernels`, with its own tensor layout. |

## What each run produces

Every script follows the same shape: warm up, profile 5 steps with `schedule(wait=1, warmup=1, active=3, repeat=1)`, then return the results.

```
traces/04_c_sdpa_attention/
  8_16_1024_64_flash_eager.json.gz   # trace
  8_16_1024_64_flash_eager.txt       # key_averages() table, sorted by CUDA time
```

The filename tag encodes the run's parameters, so repeated runs with different flags accumulate side by side instead of overwriting each other. To view the trace, open [ui.perfetto.dev](https://ui.perfetto.dev) and load the `.json.gz` file directly.

## License

MIT — see [LICENSE](LICENSE).
