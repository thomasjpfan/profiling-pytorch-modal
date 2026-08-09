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
benchmark_name = "01_matmul_add"


@app.function(image=image, gpu="A100")
def benchmark(
    size: int, dtype: Literal["bf16", "fp32"], compile: bool, warmup: bool
) -> dict[str, bytes]:
    import torch

    print(
        f"starting benchmark: size={size} dtype={dtype} "
        f"compile={compile} warmup={warmup}"
    )

    device = "cuda"
    dtype = torch.bfloat16 if dtype == "bf16" else torch.float32

    x = torch.randn(size, size, device=device, dtype=dtype)
    w = torch.randn(size, size, device=device, dtype=dtype)
    b = torch.randn(size, size, device=device, dtype=dtype)

    def fn(x, w, b):
        return torch.add(torch.matmul(x, w), b)

    fn = torch.compile(fn) if compile else fn

    def step():
        with torch.profiler.record_function("matmul_add"):
            return fn(x, w, b)

    if warmup:
        for _ in range(3):
            step()
        torch.cuda.synchronize()

    compile_tag = "compile" if compile else "eager"
    warmup_tag = "warm" if warmup else "cold"
    tag = f"{size}_{dtype}_{warmup_tag}_{compile_tag}"
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
    size: int = 64,
    dtype: Literal["bf16", "fp32"] = "bf16",
    compile: bool = True,
    warmup: bool = True,
):
    files: dict[str, bytes] = benchmark.remote(
        size=size, dtype=dtype, compile=compile, warmup=warmup
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
