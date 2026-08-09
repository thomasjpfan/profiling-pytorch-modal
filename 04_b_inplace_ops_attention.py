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
benchmark_name = "04_b_inplace_ops_attention"


@app.function(image=image, gpu="A100")
def benchmark(
    batch: int, heads: int, seq: int, head_dim: int, compile: bool
) -> dict[str, bytes]:
    import math

    import torch
    from torch import nn

    print(
        f"starting benchmark: batch={batch} heads={heads} seq={seq} "
        f"head_dim={head_dim} compile={compile}"
    )

    class NaiveCausalAttention(nn.Module):
        """softmax(QK^T / sqrt(d) + mask) @ V."""

        def __init__(self, head_dim):
            super().__init__()
            self.scale = 1.0 / math.sqrt(head_dim)

        def forward(self, q, k, v, mask):
            # q, k, v: [batch, heads, seq, head_dim]
            scores = torch.matmul(q, k.transpose(-2, -1))  # [batch, heads, seq, seq]
            scores = scores * self.scale
            scores.masked_fill_(mask, float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)  # [batch, heads, seq, head_dim]
            return out

    device = "cuda"
    dtype = torch.bfloat16

    shape = (batch, heads, seq, head_dim)
    q = torch.randn(shape, device=device, dtype=dtype)
    k = torch.randn(shape, device=device, dtype=dtype)
    v = torch.randn(shape, device=device, dtype=dtype)

    # causal mask built once
    mask = torch.triu(
        torch.ones(seq, seq, device=device, dtype=torch.bool),
        diagonal=1,
    )

    attn = NaiveCausalAttention(head_dim).to(device, dtype=dtype)
    attn.eval()

    fwd = torch.compile(attn) if compile else attn

    def step():
        with torch.profiler.record_function("attn_fwd"), torch.no_grad():
            return fwd(q, k, v, mask)

    # warmup
    for _ in range(3):
        step()
    torch.cuda.synchronize()

    compile_tag = "compile" if compile else "eager"
    tag = f"{batch}_{heads}_{seq}_{head_dim}_{compile_tag}"
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
    compile: bool = True,
):
    files: dict[str, bytes] = benchmark.remote(
        batch=batch, heads=heads, seq=seq, head_dim=head_dim, compile=compile
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
