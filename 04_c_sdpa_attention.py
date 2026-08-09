# /// script
# dependencies = [
#   "modal",
# ]
# ///
import gzip
import tempfile
from pathlib import Path
from typing import Literal

import modal

app = modal.App()
image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_pip_install("torch==2.13.0", "numpy")
    .env({"TORCHINDUCTOR_COMPILE_THREADS": "1"})
)
benchmark_name = "04_c_sdpa_attention"


@app.function(image=image, gpu="A100")
def benchmark(
    batch: int,
    heads: int,
    seq: int,
    head_dim: int,
    # "auto" lets SDPA pick the backend; the rest force a single backend
    backend: Literal["auto", "math", "flash", "efficient", "cudnn"],
    compile: bool,
) -> dict[str, bytes]:
    import torch
    from torch.nn import functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    print(
        f"starting benchmark: batch={batch} heads={heads} seq={seq} "
        f"head_dim={head_dim} backend={backend} compile={compile}"
    )

    # the backends torch.nn.functional.scaled_dot_product_attention can dispatch to
    backends = {
        "math": SDPBackend.MATH,
        "flash": SDPBackend.FLASH_ATTENTION,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
        "cudnn": SDPBackend.CUDNN_ATTENTION,
    }

    device = "cuda"
    dtype = torch.bfloat16

    shape = (batch, heads, seq, head_dim)
    q = torch.randn(shape, device=device, dtype=dtype)
    k = torch.randn(shape, device=device, dtype=dtype)
    v = torch.randn(shape, device=device, dtype=dtype)

    def attn(q, k, v):
        # is_causal=True asks SDPA to apply the causal mask internally,
        # so we never build or materialize a [seq, seq] mask ourselves.
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    fwd = torch.compile(attn) if compile else attn

    def step():
        with torch.profiler.record_function("sdpa_fwd"), torch.no_grad():
            if backend == "auto":
                return fwd(q, k, v)
            with sdpa_kernel(backends[backend]):
                return fwd(q, k, v)

    # warmup
    for _ in range(3):
        step()
    torch.cuda.synchronize()

    compile_tag = "compile" if compile else "eager"
    tag = f"{batch}_{heads}_{seq}_{head_dim}_{backend}_{compile_tag}"
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
    backend: Literal["auto", "math", "flash", "efficient", "cudnn"] = "auto",
    compile: bool = True,
):
    files: dict[str, bytes] = benchmark.remote(
        batch=batch,
        heads=heads,
        seq=seq,
        head_dim=head_dim,
        backend=backend,
        compile=compile,
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
