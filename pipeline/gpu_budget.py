"""Hard PyTorch allocation cap plus observed peak for a single-model process."""

from __future__ import annotations


def configure(torch, device: str, limit_mb: int) -> None:
    if device == "cpu":
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but this PyTorch installation has no CUDA support")
    total = torch.cuda.get_device_properties(0).total_memory
    limit = limit_mb * 1024 * 1024
    if limit > total:
        raise ValueError(f"VRAM limit {limit_mb} MB exceeds device capacity")
    torch.cuda.set_per_process_memory_fraction(limit / total, device=0)
    torch.cuda.reset_peak_memory_stats(0)


def check(torch, device: str, limit_mb: int) -> float:
    if device == "cpu":
        return 0.0
    torch.cuda.synchronize(0)
    peak = torch.cuda.max_memory_reserved(0) / (1024 * 1024)
    if peak >= limit_mb:
        raise MemoryError(f"PyTorch peak reserved VRAM {peak:.0f} MB reached {limit_mb} MB limit")
    return float(peak)
