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
    .uv_pip_install("torch==2.13.0", "numpy")
    .env({"TORCHINDUCTOR_COMPILE_THREADS": "1"})
)
benchmark_name = "02_linear"


@app.function(image=image, gpu="A100")
def benchmark(batch: int, in_dim: int, out_dim: int, compile: bool) -> dict[str, bytes]:
    import torch
    from torch import nn

    print(
        f"starting benchmark: batch={batch} in_dim={in_dim} "
        f"out_dim={out_dim} compile={compile}"
    )

    device = "cuda"
    dtype = torch.bfloat16

    x = torch.randn(batch, in_dim, device=device, dtype=dtype)
    linear_layer = nn.Linear(in_dim, out_dim, bias=True).to(device, dtype=dtype)
    linear_layer.eval()

    print(linear_layer.weight.shape)
    print(linear_layer.bias.shape)

    fwd = torch.compile(linear_layer) if compile else linear_layer

    def step():
        with torch.profiler.record_function("linear_fwd"), torch.no_grad():
            return fwd(x)

    # warmup
    for _ in range(3):
        step()
    torch.cuda.synchronize()

    compile_tag = "compile" if compile else "eager"
    tag = f"{batch}_{in_dim}_{out_dim}_{compile_tag}"
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
    batch: int = 1024,
    in_dim: int = 32,
    out_dim: int = 64,
    compile: bool = True,
):
    files: dict[str, bytes] = benchmark.remote(
        batch=batch, in_dim=in_dim, out_dim=out_dim, compile=compile
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
