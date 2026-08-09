# /// script
# dependencies = [
#   "modal",
# ]
# ///
import gzip
import tempfile
from pathlib import Path

import modal

app = modal.App()
image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_pip_install("torch==2.13.0", "numpy", "kernels==0.16.0")
    .env({"TORCHINDUCTOR_COMPILE_THREADS": "1", "HF_HOME": "/hf_home"})
)
hf_home = modal.Volume.from_name("hf-cache", create_if_missing=True)
benchmark_name = "04_d_kernels_attention"


@app.function(image=image, gpu="A100", volumes={"/hf_home": hf_home})
def benchmark(batch: int, heads: int, seq: int, head_dim: int) -> dict[str, bytes]:
    import torch
    from kernels import get_kernel

    print(
        f"starting benchmark: batch={batch} heads={heads} seq={seq} head_dim={head_dim}"
    )

    device = "cuda"
    dtype = torch.bfloat16

    # flash-attn expects [batch, seq, heads, head_dim] (seq and heads are
    # swapped compared to SDPA's [batch, heads, seq, head_dim]).
    shape = (batch, seq, heads, head_dim)
    q = torch.randn(shape, device=device, dtype=dtype)
    k = torch.randn(shape, device=device, dtype=dtype)
    v = torch.randn(shape, device=device, dtype=dtype)

    # pre-built, version-pinned FlashAttention kernel from the Hugging Face Hub
    flash = get_kernel("kernels-community/flash-attn2", version=3)

    def attn(q, k, v):
        return flash.flash_attn_func(q, k, v, causal=True)

    def step():
        with torch.profiler.record_function("flash_kernel_fwd"), torch.no_grad():
            return attn(q, k, v)

    # warmup
    for _ in range(3):
        step()
    torch.cuda.synchronize()

    tag = f"{batch}_{heads}_{seq}_{head_dim}_flashattn"
    print(f"tag={tag}")

    schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=schedule,
        record_shapes=False,  # adds CPU overhead
        profile_memory=False,  # adds CPU overhead
        with_stack=False,
    ) as prof:
        for _ in range(5):
            step()
            prof.step()

    torch.cuda.synchronize()
    print(f"peak memory {torch.cuda.max_memory_allocated() / 1024**2:.2f} MiB")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_trace = Path(tmp_dir) / f"{tag}.json.gz"
        prof.export_chrome_trace(str(tmp_trace))
        trace_gz = tmp_trace.read_bytes()

    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=15)

    return {
        f"{tag}.json.gz": trace_gz,
        f"{tag}.txt.gz": gzip.compress(table.encode()),
    }


@app.local_entrypoint()
def main(
    batch: int = 8,
    heads: int = 16,
    seq: int = 1024,
    head_dim: int = 64,
):
    files: dict[str, bytes] = benchmark.remote(
        batch=batch, heads=heads, seq=seq, head_dim=head_dim
    )

    trace_root = Path.cwd() / "traces" / benchmark_name
    trace_root.mkdir(parents=True, exist_ok=True)

    table = ""
    for name, data in files.items():
        if name.endswith(".txt.gz"):
            name, data = name.removesuffix(".gz"), gzip.decompress(data)
            table = data.decode()
        dst_path = trace_root / name
        dst_path.write_bytes(data)
        print(f"wrote {dst_path} ({len(data) / 1024**2:.2f} MiB)")

    print(table)
