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
benchmark_name = "03_kernels_mlp"
hf_home = modal.Volume.from_name("hf-cache", create_if_missing=True)


@app.function(image=image, gpu="A100", volumes={"/hf_home": hf_home})
def benchmark(
    batch: int, seq: int, dim: int, hidden: int, compile: bool
) -> dict[str, bytes]:
    import torch
    from kernels import get_kernel

    print(
        f"starting benchmark: batch={batch} seq={seq} dim={dim} "
        f"hidden={hidden} compile={compile}"
    )

    device = "cuda"
    dtype = torch.bfloat16

    x = torch.randn(batch, seq, dim, device=device, dtype=dtype)

    class Config:
        hidden_size = dim
        intermediate_size = hidden
        hidden_act = "gelu_pytorch_tanh"

    # pre-built, version-pinned Liger kernels from the Hugging Face Hub
    kernels_layers = get_kernel("kernels-community/liger-kernels", version=3).layers
    kernels_geglu_mlp = kernels_layers.LigerGEGLUMLP
    kernels_geglu_mlp = (
        kernels_geglu_mlp(Config()).to(device=device, dtype=dtype).eval()
    )

    fwd = torch.compile(kernels_geglu_mlp) if compile else kernels_geglu_mlp

    def step():
        with torch.profiler.record_function("kernels_mlp_fwd"), torch.no_grad():
            return fwd(x)

    # warmup
    for _ in range(3):
        step()
    torch.cuda.synchronize()

    compile_tag = "compile" if compile else "eager"
    tag = f"{batch}_{seq}_{dim}_{hidden}_{compile_tag}"
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
    batch: int = 64,
    seq: int = 128,
    dim: int = 768,
    hidden: int = 3072,
    compile: bool = True,
):
    files: dict[str, bytes] = benchmark.remote(
        batch=batch, seq=seq, dim=dim, hidden=hidden, compile=compile
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
