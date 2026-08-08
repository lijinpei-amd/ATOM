"""Repair the ragged tail the gfx1250 gluon fp8_mqa_logits kernel skips.

That kernel is numerically exact (cos 1.000000000 vs a float32 replay) on every
column it writes, but for a query row whose window starts at a nonzero offset it
stops short of cu_end and leaves the remainder at the -inf fill. The generic
triton kernel is NOT a usable fallback on gfx1250 -- it covers the whole window
but returns garbage (cos 0.117 vs the same reference).

So keep the gluon kernel and recompute only what it left behind, replaying the
kernel's own arithmetic:

    logits[r, c] = sum_h relu( (q[r,h,:] . kv[c,:]) * kv_scale[c] ) * w[r,h]

The skipped region is found from the -inf fill itself rather than assumed to be
a fixed tile multiple, so it stays correct whatever the kernel's blocking is.
"""
import torch


def repair_tail(logits, Q, KV, kv_scales, weights, cu_starts, cu_ends,
                row_chunk=256):
    """In-place fill of every in-window column the kernel left non-finite."""
    n_rows, n_cols = logits.shape
    dev = logits.device
    cols = torch.arange(n_cols, device=dev)
    cs = cu_starts[:n_rows].long()
    ce = cu_ends[:n_rows].long().clamp(max=n_cols)
    in_win = (cols.unsqueeze(0) >= cs.unsqueeze(1)) & (
        cols.unsqueeze(0) < ce.unsqueeze(1))
    missing = in_win & ~torch.isfinite(logits)
    any_missing = missing.any(1)
    if not bool(any_missing.any()):
        return logits
    # first missing column per row; rows with none get an empty range
    t0 = torch.where(any_missing, missing.float().argmax(1), ce)
    n_missing = (ce - t0).clamp_min(0)
    span = int(n_missing.max())
    if span == 0:
        return logits

    ks_all = kv_scales.reshape(-1).float()  # ATOM passes [total_kv, 1]
    qf_all = Q.float()
    kvf_all = KV.float()
    off = torch.arange(span, device=dev)
    for lo in range(0, n_rows, row_chunk):
        hi = min(lo + row_chunk, n_rows)
        m = n_missing[lo:hi]
        if int(m.max()) == 0:
            continue
        c = t0[lo:hi].unsqueeze(1) + off.unsqueeze(0)          # [R, span]
        valid = off.unsqueeze(0) < m.unsqueeze(1)
        safe = c.clamp(0, n_cols - 1)
        s = torch.einsum("rhd,rcd->rhc", qf_all[lo:hi], kvf_all[safe])
        s = (s * ks_all[safe].unsqueeze(1)).clamp_min(0.0)
        tail = (s * weights[lo:hi].unsqueeze(-1)).sum(1)       # [R, span]
        rows = torch.arange(lo, hi, device=dev).unsqueeze(1).expand_as(safe)
        logits[rows[valid], safe[valid]] = tail[valid]
    return logits
