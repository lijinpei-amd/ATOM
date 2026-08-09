"""gfx1250 kernel-discovery shim.

Loaded via PYTHONPATH so it also runs inside ATOM's spawned worker processes.

Two jobs:
  1. Prefer aiter's Triton implementations over the CK/HIP ones wherever a
     same-named callable exists under ``aiter.ops.triton`` (CK-tile does not
     compile for gfx1250 at all).
  2. For any CK/HIP op that survives step 1 and then fails to build, record it
     in a JSONL report instead of letting an opaque compiler dump be the only
     evidence.

Nothing here is loaded unless GFX1250_SHIM=1.
"""

import os
import sys

# Preserve Debian's stock sitecustomize behaviour, which we shadow.
try:
    import apport_python_hook
except ImportError:
    pass
else:
    apport_python_hook.install()

_ENABLED = os.environ.get("GFX1250_SHIM") == "1"
_LOG_PATH = os.environ.get("GFX1250_SHIM_LOG", "/tmp/gfx1250_shim.jsonl")
_AUTO_TRITON = os.environ.get("GFX1250_SHIM_AUTO_TRITON", "1") == "1"


def _emit(record):
    import json
    import threading

    record["pid"] = os.getpid()
    line = json.dumps(record, default=str)
    # append is atomic enough for short lines across our few processes
    with open(_LOG_PATH, "a") as fh:
        fh.write(line + "\n")
    print("[gfx1250-shim] " + line, file=sys.stderr, flush=True)


class Gfx1250MissingKernel(RuntimeError):
    """Raised when an aiter CK/HIP op has no gfx1250 build and no Triton peer."""


# --------------------------------------------------------------------------
# 1. Triton preference
# --------------------------------------------------------------------------

# Generic names that collide across unrelated modules and must never be
# auto-redirected on a name match alone.
_DENY = {
    "tanh", "exp", "log", "sigmoid", "silu", "gelu", "softmax",
    "get_dtype_max", "get_dtype_min", "arch_info", "is_hip",
}


# CK/HIP op name -> (triton module, triton callable) where the two projects
# spell the same operation differently, so the name matcher can't pair them.
_ALIASES = {
    "rmsnorm2d_fwd": ("aiter.ops.triton.normalization.rmsnorm", "rms_norm"),
    "layernorm2d_fwd": ("aiter.ops.triton.normalization.norm", "layer_norm"),
}


# --------------------------------------------------------------------------
# GLM_COVERAGE=<path>: which replacements does the model actually call?
#
# The shim patches everything it can find a peer for; that is a much larger set
# than the model touches. Wrapping the table at the one point where it is
# assembled separates the two: one 4-question run showed 15 of ~55 live, and
# nothing else.
#
# Caveat: a later _install_* block that rebinds the same name shadows the
# wrapper, so its op reads as 0 calls. GLM_TRITON_SPARSE_MLA does this to
# mla_prefill_asm_fwd -- cross-check against the "[tsm]" stderr lines.
#
# Each process appends one JSON record at exit; sum "hits" across them.
# --------------------------------------------------------------------------
_COV = os.environ.get("GLM_COVERAGE")
_COV_HITS = {}
_COV_ORIGIN = {}


def _cov_wrap(op, impl):
    import functools

    def w(*a, **kw):
        _COV_HITS[op] = _COV_HITS.get(op, 0) + 1
        return impl(*a, **kw)

    try:
        functools.update_wrapper(w, impl)
    except Exception:
        pass
    return w


def _cov_dump():
    import json
    if not _COV_HITS and not _COV_ORIGIN:
        return
    rec = {"pid": os.getpid(),
           "hits": _COV_HITS,
           "origin": _COV_ORIGIN}
    with open(_COV, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


if _COV:
    import atexit
    atexit.register(_cov_dump)


def _make_custom_impls():
    """Hand-written triton adapters for ops whose triton peer differs in layout.

    Returns {aiter top-level name: callable}. Built lazily so importing this
    module never drags in torch.
    """
    import torch  # noqa: F401

    from aiter.ops.triton.rope.rope import rope_cached_thd_positions_2c_fwd_inplace

    def rope_cached_positions_2c_fwd_inplace(
        input_x, input_y, cos, sin, positions,
        rotate_style, reuse_freqs_front_part, nope_first,
    ):
        # CK takes (s, b, h, d) with s == 1 from ATOM; triton wants THD (t, h, d).
        # squeeze(0) is a view, so the in-place write lands in the caller's tensors.
        # ATOM's cos/sin are (max_pos, 1, 1, d') but the triton kernel unpacks
        # exactly two cos strides, so the extra dims must be collapsed first or
        # every later positional arg shifts.
        rope_cached_thd_positions_2c_fwd_inplace(
            input_x.squeeze(0),
            input_y.squeeze(0),
            cos.reshape(cos.shape[0], cos.shape[-1]),
            sin.reshape(sin.shape[0], sin.shape[-1]),
            positions.reshape(-1),
            rotate_style,
            reuse_freqs_front_part,
            nope_first,
        )

    from aiter.ops.triton.quant.quant import dynamic_per_token_quant_fp8_i8

    def dynamic_per_token_scaled_quant(
        out, input, scales, scale_ub=None, shuffle_scale=False,
        num_rows=None, num_rows_factor=1,
    ):
        # Triton peer covers only the plain per-token case; anything else would
        # silently compute the wrong thing, so surface it instead.
        if scale_ub is not None or shuffle_scale or num_rows is not None:
            raise Gfx1250MissingKernel(
                "dynamic_per_token_scaled_quant: triton peer does not support "
                f"scale_ub={scale_ub is not None} shuffle_scale={shuffle_scale} "
                f"num_rows={num_rows is not None}"
            )
        dynamic_per_token_quant_fp8_i8(
            out.reshape(-1, out.shape[-1]),
            input.reshape(-1, input.shape[-1]),
            scales.reshape(-1),
        )

    from aiter.ops.triton.activation import fused_silu_mul

    def silu_and_mul(out, input, limit=0.0):
        if limit:
            raise Gfx1250MissingKernel(
                f"silu_and_mul: triton peer has no clamp support (limit={limit})"
            )
        fused_silu_mul(input, out)

    def indexer_k_quant_and_cache(
        k, kv_cache, slot_mapping, quant_block_size, scale_fmt, preshuffle=False,
    ):
        """Torch replica of csrc/kernels/cache_kernels.cu::indexer_k_quant_and_cache.

        Per-quant-block fp8 quantisation of the DSA indexer keys, written into the
        paged cache. Values live in the first ``cache_block_size * head_dim`` bytes
        of each page (optionally in the MFMA 16x16 preshuffled order); the fp32
        scales follow immediately after, always in unshuffled order.
        """
        num_tokens = min(k.shape[0], slot_mapping.shape[0])
        head_dim = k.shape[1]
        cache_block_size = kv_cache.shape[1]
        cache_stride = kv_cache.shape[2]
        Q = int(quant_block_size)
        n_q = head_dim // Q

        slots = slot_mapping[:num_tokens].to(torch.int64)
        if slots.numel() == 0:
            return
        # The HIP kernel early-returns on slot < 0 (padded tokens). Boolean-mask
        # compaction would be a device->host sync, which is illegal inside a
        # CUDA-graph capture, so instead every padded row is rewritten to be an
        # exact duplicate of the first valid row: same destination, same bytes,
        # so the redundant writes are order-independent no-ops. (Degenerate case:
        # a batch with no valid slot at all writes one junk slot into block 0.)
        valid = slots >= 0
        rows = torch.arange(slots.shape[0], device=slots.device)
        ref = torch.argmax(valid.to(torch.uint8))
        src = torch.where(valid, rows, ref)
        slots = slots.clamp_min(0)[src]
        kf = k[:num_tokens].float()[src]
        t = slots.shape[0]

        FP8_MAX = 448.0
        blocks = kf.view(t, n_q, Q)
        scale = blocks.abs().amax(-1).clamp_min(1e-4) / FP8_MAX
        if scale_fmt == "ue8m0":
            scale = torch.exp2(torch.ceil(torch.log2(scale)))
        q8 = (
            (blocks / scale.unsqueeze(-1))
            .clamp(-FP8_MAX, FP8_MAX)
            .to(torch.float8_e4m3fn)
            .view(t, head_dim)
            .view(torch.uint8)
        )

        dev = k.device
        block_id = slots // cache_block_size
        off = (slots % cache_block_size).unsqueeze(1)
        base = (block_id * (cache_block_size * cache_stride)).unsqueeze(1)
        d = torch.arange(head_dim, device=dev, dtype=torch.int64).unsqueeze(0)

        if preshuffle:
            TILE = 16
            idx = (
                base
                + (off // TILE) * (TILE * head_dim)
                + (d // TILE) * (TILE * TILE)
                + (off % TILE) * TILE
                + (d % TILE)
            )
        else:
            idx = base + off * head_dim + d

        flat = kv_cache.view(-1)
        u8 = flat if flat.dtype == torch.uint8 else flat.view(torch.uint8)
        u8[idx.reshape(-1)] = q8.reshape(-1)

        d_q = torch.arange(n_q, device=dev, dtype=torch.int64).unsqueeze(0) * Q
        s_idx = (
            base + cache_block_size * head_dim + ((off * head_dim + d_q) * 4) // Q
        ) // 4
        u8.view(torch.float32)[s_idx.reshape(-1)] = scale.reshape(-1)

    from aiter.ops.triton.attention.mha import (
        flash_attn_varlen_func as _triton_flash_attn_varlen_func,
    )

    def flash_attn_varlen_func(*args, min_seqlen_q=None, **kwargs):
        # CK takes min_seqlen_q as a varlen-scheduler hint; it has no effect on
        # the result, and the triton kernel has no equivalent knob.
        return _triton_flash_attn_varlen_func(*args, **kwargs)

    def mla_prefill_asm_fwd(
        q, kv_buffer, qo_indptr, kv_indptr, kv_indices, kv_last_page_lens,
        max_seqlen_q, sm_scale, logits, attn_lse, *args, **kwargs,
    ):
        """Torch reference for the paged sparse-MLA prefill attention.

        Each query group b attends to exactly the KV slots listed in
        kv_indices[kv_indptr[b]:kv_indptr[b+1]] — an arbitrary per-token gather
        produced by the DSA top-k, which is why the block-table-based gluon
        mla_prefill_fwd is not a drop-in replacement.

        Vectorised for the shape ATOM's sparse path actually uses (page_size 1,
        one query per group); other shapes fall back to a per-group loop.
        """
        _mi = os.environ.get("GLM_DUMP_MLAIDX")
        if _mi and not getattr(mla_prefill_asm_fwd, "_dumped", False):
            mla_prefill_asm_fwd._dumped = True
            import numpy as _np
            _n = min(int(qo_indptr.shape[0]) - 1, 4096)
            _np.savez(_mi,
                      qo_indptr=qo_indptr[: _n + 1].detach().cpu().numpy(),
                      kv_indptr=kv_indptr[: _n + 1].detach().cpu().numpy(),
                      kv_indices=kv_indices[:200000].detach().cpu().numpy(),
                      kv_last_page_lens=kv_last_page_lens[:_n].detach().cpu().numpy(),
                      kv_shape=_np.array(kv_buffer.shape),
                      q=q[: min(4, q.shape[0])].detach().float().cpu().numpy())
            print(f"[dump] mla idx -> {_mi}", file=sys.stderr, flush=True)
        num_page, page_size, _n_kv, qk_head_dim = kv_buffer.shape
        v_head_dim = logits.shape[-1]
        kv_flat = kv_buffer.reshape(num_page * page_size, qk_head_dim)
        out = logits.reshape(-1, logits.shape[-2], v_head_dim)
        lse = attn_lse.reshape(-1, attn_lse.shape[-2])
        qo = qo_indptr.to(torch.int64)
        kvp = kv_indptr.to(torch.int64)
        n_groups = qo.shape[0] - 1

        uniform_single_q = page_size == 1 and int(max_seqlen_q) == 1

        def attend(q_rows, idx, mask):
            """q_rows [g, H, D], idx [g, K] int64, mask [g, K] bool -> writes out/lse."""
            k = kv_flat[idx.reshape(-1)].view(*idx.shape, qk_head_dim).float()
            s = torch.einsum("ghd,gkd->ghk", q_rows.float(), k) * sm_scale
            s = s.masked_fill(~mask.unsqueeze(1), float("-inf"))
            m = s.amax(-1, keepdim=True)
            m = torch.where(torch.isneginf(m), torch.zeros_like(m), m)
            e = torch.exp(s - m)
            denom = e.sum(-1, keepdim=True)
            p = e / denom.clamp_min(1e-30)
            o = torch.einsum("ghk,gkv->ghv", p, k[..., :v_head_dim])
            return o, (m.squeeze(-1) + torch.log(denom.squeeze(-1).clamp_min(1e-30)))

        if uniform_single_q:
            counts = (kvp[1:] - kvp[:-1]).clamp_min(0)
            max_k = int(counts.max().item()) if n_groups else 0
            if max_k == 0:
                out.zero_()
                lse.zero_()
                return
            ar = torch.arange(max_k, device=q.device, dtype=torch.int64)
            # 4096 queries x 2048 KV x 576 dims does not fit as one gather.
            chunk = max(1, min(n_groups, (1 << 26) // max(max_k * qk_head_dim, 1)))
            for lo in range(0, n_groups, chunk):
                hi = min(lo + chunk, n_groups)
                cnt = counts[lo:hi].unsqueeze(1)
                flat = kvp[lo:hi].unsqueeze(1) + ar.unsqueeze(0)
                idx = kv_indices.to(torch.int64)[flat.clamp(0, kv_indices.numel() - 1)]
                mask = (ar.unsqueeze(0) < cnt) & (idx >= 0)
                o, l = attend(q[qo[lo] : qo[hi]], idx.clamp_min(0), mask)
                out[lo:hi] = o.to(out.dtype)
                lse[lo:hi] = l
            return

        for b in range(n_groups):
            q_rows = q[qo[b] : qo[b + 1]]
            pages = kv_indices.to(torch.int64)[kvp[b] : kvp[b + 1]]
            if pages.numel() == 0 or q_rows.shape[0] == 0:
                out[qo[b] : qo[b + 1]].zero_()
                continue
            slots = (pages.unsqueeze(1) * page_size + torch.arange(
                page_size, device=q.device, dtype=torch.int64
            ).unsqueeze(0)).reshape(-1)
            keep = torch.ones(slots.shape[0], dtype=torch.bool, device=q.device)
            last = int(kv_last_page_lens[b].item()) if kv_last_page_lens is not None else page_size
            if last < page_size:
                keep[-(page_size - last):] = False
            n_q = q_rows.shape[0]
            idx = slots.unsqueeze(0).expand(n_q, -1)
            mask = keep.unsqueeze(0).expand(n_q, -1)
            o, l = attend(q_rows, idx, mask)
            out[qo[b] : qo[b + 1]] = o.to(out.dtype)
            lse[qo[b] : qo[b + 1]] = l

    def cp_gather_indexer_k_quant_cache(
        kv_cache, dst_k, dst_scale, block_table, cu_seq_lens, preshuffle=False,
    ):
        """Inverse of indexer_k_quant_and_cache: paged cache -> contiguous [T, D].

        Reverses the same MFMA 16x16 tile mapping and copies the fp32 scales that
        follow the value region of each page.
        """
        num_tokens, head_dim = dst_k.shape
        n_q = dst_scale.shape[1] * dst_scale.element_size() // 4
        Q = head_dim // n_q
        cache_block_size = kv_cache.shape[1]
        cache_stride = kv_cache.shape[2]

        dev = dst_k.device
        tok = torch.arange(num_tokens, device=dev, dtype=torch.int64)
        cs = cu_seq_lens.to(torch.int64)
        # cu_seq_lens is [bs+1]; token t belongs to the batch whose span covers it.
        b = (torch.searchsorted(cs, tok, right=True) - 1).clamp(
            0, block_table.shape[0] - 1
        )
        in_batch = tok - cs[b]
        # ATOM pads block_tables with -1 when it is shorter than num_prefills.
        blk = block_table.to(torch.int64)[
            b, (in_batch // cache_block_size).clamp(0, block_table.shape[1] - 1)
        ].clamp_min(0)
        off = (in_batch % cache_block_size).unsqueeze(1)
        base = (blk * (cache_block_size * cache_stride)).unsqueeze(1)
        d = torch.arange(head_dim, device=dev, dtype=torch.int64).unsqueeze(0)

        if preshuffle:
            TILE = 16
            src = (
                base
                + (off // TILE) * (TILE * head_dim)
                + (d // TILE) * (TILE * TILE)
                + (off % TILE) * TILE
                + (d % TILE)
            )
        else:
            src = base + off * head_dim + d

        flat = kv_cache.reshape(-1)
        u8 = flat if flat.dtype == torch.uint8 else flat.view(torch.uint8)
        dst_k.view(torch.uint8).copy_(u8[src.reshape(-1)].view(num_tokens, head_dim))

        d_q = torch.arange(n_q, device=dev, dtype=torch.int64).unsqueeze(0) * Q
        s_src = (
            base + cache_block_size * head_dim + ((off * head_dim + d_q) * 4) // Q
        ) // 4
        scales = u8.view(torch.float32)[s_src.reshape(-1)].view(num_tokens, n_q)
        dst_scale.reshape(-1).view(torch.float32).view(num_tokens, n_q).copy_(scales)

    def get_mla_metadata_v1(
        seqlens_qo_indptr, seqlens_kv_indptr, kv_last_page_lens,
        num_heads_per_head_k, num_heads_k, is_causal,
        work_metadata_ptrs, work_info_set, work_indptr,
        reduce_indptr, reduce_final_map, reduce_partial_map,
        *args, **kwargs,
    ):
        """No-work stub for the CK/HIP sparse-MLA work scheduler.

        ATOM builds this schedule unconditionally in prepare_prefill, but the
        only consumer is mla_decode_fwd on the fp8 KV-cache path; with a bf16
        KV cache the forward goes to mla_prefill_fwd, which ignores it. Zeroed
        indptrs encode "no work items", so a consumer that did read them would
        do nothing rather than chase garbage.
        """
        for buf in (
            work_metadata_ptrs, work_info_set, work_indptr,
            reduce_indptr, reduce_final_map, reduce_partial_map,
        ):
            if isinstance(buf, torch.Tensor):
                buf.zero_()

    _TOPK_IDX_BITS = 21          # column index field; covers 2M columns

    def _ordered_topk(scores, kk):
        """Top-kk columns per row, ordered by (score desc, column asc).

        Packs an order-preserving map of the float bits with the column index
        into one int64 key, so a single topk gives both the ordering and the
        tie-break without a full sort. Ties go to the smaller column, and
        columns run in token order within a row window, so that is token order.
        """
        w = scores.shape[1]
        if w >= (1 << _TOPK_IDX_BITS):
            vals, idx = torch.sort(scores, dim=1, descending=True, stable=True)
            return vals[:, :kk], idx[:, :kk]
        key = scores.contiguous().view(torch.int32).to(torch.int64)
        key = torch.where(key == -(1 << 31), torch.zeros_like(key), key)  # -0.0 -> +0.0
        key = torch.where(key >= 0, key + (1 << 31), ~key)
        key <<= _TOPK_IDX_BITS
        key -= torch.arange(w, device=scores.device, dtype=torch.int64)
        _, idx = torch.topk(key, kk, dim=1)
        del key
        return scores.gather(1, idx), idx

    def _row_topk_write(scores, indices_out, k, values_out=None):
        """Shared tail of the two top_k_per_row ops.

        ``scores`` already has out-of-range columns set to -inf. The C++ kernels
        write exactly ``indices_out.shape[1]`` ints per row: the selected column
        indices followed by -1 sentinels, in no particular order.
        """
        out_k = indices_out.shape[1]
        kk = min(int(k), out_k, scores.shape[1])
        if os.environ.get("GLM_ORDERED_TOPK", "1") == "1":
            vals, idx = _ordered_topk(scores, kk)
        else:
            vals, idx = torch.topk(scores, kk, dim=1)
        idx = torch.where(torch.isneginf(vals), torch.full_like(idx, -1), idx)
        indices_out[:, :kk] = idx.to(indices_out.dtype)
        if out_k > kk:
            indices_out[:, kk:] = -1
        if values_out is not None:
            values_out[:, :kk] = vals.to(values_out.dtype)
            if values_out.shape[1] > kk:
                values_out[:, kk:] = float("-inf")

    def concat_and_cache_mla(kv_c, k_pe, kv_cache, slot_mapping,
                             kv_cache_dtype=None, scale=None, *a, **kw):
        """kv_cache[slot] = cat(kv_c, k_pe), for each token's slot.

        kv_cache arrives as [num_blocks, block_size, d] (ATOM re-views the flat
        pool that way); slot_mapping indexes it as one flat run of slots, which
        is exactly what the row-major flatten gives. Negative slots mark padded
        tokens and are skipped.
        """
        n = slot_mapping.numel()
        d = kv_cache.shape[-1]
        flat = kv_cache.reshape(-1, d)
        val = torch.cat([kv_c.reshape(n, -1), k_pe.reshape(n, -1)], dim=-1)
        slots = slot_mapping.reshape(-1).long()
        keep = slots >= 0
        if flat.dtype.itemsize == 1 and scale is not None:
            # fp8 pool: the kernel divides by the static scale before storing
            val = (val.float() / scale.float().reshape(())).to(flat.dtype)
        else:
            val = val.to(flat.dtype)
        flat[slots[keep]] = val[keep]

    def top_k_per_row_prefill(
        logits, rowStarts, rowEnds, indices, values, numRows, stride0, stride1,
        k=2048, workspace=None,
    ):
        n = int(numRows)
        scores = logits[:n].float()
        cols = torch.arange(scores.shape[1], device=scores.device)
        in_row = (cols.unsqueeze(0) >= rowStarts[:n].unsqueeze(1)) & (
            cols.unsqueeze(0) < rowEnds[:n].unsqueeze(1)
        )
        _row_topk_write(
            scores.masked_fill(~in_row, float("-inf")),
            indices[:n],
            k,
            None if values is None else values[:n],
        )

    def top_k_per_row_decode(
        logits, next_n, seqLens, indices, numRows, stride0, stride1,
        k=2048, workspace=None,
    ):
        n = int(numRows)
        nn = max(int(next_n), 1)
        scores = logits[:n].float()
        rows = torch.arange(n, device=scores.device)
        # Matches topk_per_row_kernels.cu: each MTP token in a batch sees one
        # more committed position than the previous one.
        row_len = seqLens.to(torch.int64)[rows // nn] - nn + (rows % nn) + 1
        cols = torch.arange(scores.shape[1], device=scores.device)
        in_row = cols.unsqueeze(0) < row_len.unsqueeze(1)
        _row_topk_write(scores.masked_fill(~in_row, float("-inf")), indices[:n], k)

    def greedy_sample(out, input):
        out.copy_(input.argmax(dim=-1).to(out.dtype))

    def mixed_sample_outer_exponential(out, input, exponentials, temperature, eps=1e-10):
        # Gumbel-max with pre-drawn Exp(1) noise: argmax(p / e) ~ Categorical(p).
        # "mixed" = rows whose temperature is 0 fall back to greedy argmax.
        logits = input.float()
        t = temperature.float().reshape(-1, 1)
        probs = torch.softmax(logits / t.clamp_min(eps), dim=-1)
        sampled = (probs / exponentials.float().clamp_min(eps)).argmax(dim=-1)
        out.copy_(
            torch.where(t.squeeze(-1) <= eps, logits.argmax(dim=-1), sampled).to(out.dtype)
        )

    return {
        "greedy_sample": greedy_sample,
        "flash_attn_varlen_func": flash_attn_varlen_func,
        "get_mla_metadata_v1": get_mla_metadata_v1,
        "_get_mla_metadata_v1_impl": get_mla_metadata_v1,
        "indexer_k_quant_and_cache": indexer_k_quant_and_cache,
        "cp_gather_indexer_k_quant_cache": cp_gather_indexer_k_quant_cache,
        "mla_prefill_asm_fwd": mla_prefill_asm_fwd,
        "concat_and_cache_mla": concat_and_cache_mla,
        "top_k_per_row_prefill": top_k_per_row_prefill,
        "_top_k_per_row_prefill": top_k_per_row_prefill,
        "top_k_per_row_decode": top_k_per_row_decode,
        "_top_k_per_row_decode": top_k_per_row_decode,
        "mixed_sample_outer_exponential": mixed_sample_outer_exponential,
        "rope_cached_positions_2c_fwd_inplace": rope_cached_positions_2c_fwd_inplace,
        "dynamic_per_token_scaled_quant": dynamic_per_token_scaled_quant,
        "silu_and_mul": silu_and_mul,
    }


def _build_triton_index():
    """name -> (callable, module) for every public callable under aiter.ops.triton."""
    import importlib
    import pkgutil

    import aiter.ops.triton as troot

    index = {}
    for mod_info in pkgutil.walk_packages(troot.__path__, troot.__name__ + "."):
        name = mod_info.name
        # _triton_kernels / _gluon_kernels hold raw @triton.jit kernels, not
        # host-side entry points; importing them is slow and pointless here.
        if "._triton_kernels" in name or "._gluon_kernels" in name:
            continue
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        for attr in dir(mod):
            if attr.startswith("_"):
                continue
            obj = getattr(mod, attr, None)
            if not callable(obj) or attr in _DENY:
                continue
            if getattr(obj, "__module__", "") != name:
                continue  # only definitions, not re-exports
            # Raw @triton.jit / @gluon.jit kernels are device-side; they are not
            # drop-in replacements for a host-side aiter entry point.
            if type(obj).__name__ in ("JITFunction", "GluonJITFunction"):
                continue
            index.setdefault(attr, (obj, name))
    return index


def _install_topk_dump(path):
    """Save the first prefill top-k selection, to compare index choices across machines."""
    import sys as _s
    state = {"n": 0}

    def wrap(orig):
        def wrapper(logits, rowStarts, rowEnds, indices, values, numRows,
                    stride0, stride1, k=2048, *a, **kw):
            r = orig(logits, rowStarts, rowEnds, indices, values, numRows,
                     stride0, stride1, k, *a, **kw)
            if state["n"] == 0:
                state["n"] = 1
                import numpy as np
                # Sample across the whole batch: the first rows have only a
                # handful of candidate KV positions, so their top-k is trivial.
                N = int(numRows)
                sel = list(range(max(N - 32, 0), N)) if N <= 64 else \
                    [min(int(i * (N - 1) / 31), N - 1) for i in range(32)]
                import torch as _t
                si = _t.tensor(sel, device=indices.device)
                np.savez(path,
                         rows=np.array(sel),
                         idx=indices[si].detach().cpu().numpy(),
                         logits=logits[si].detach().float().cpu().numpy(),
                         rs=rowStarts[si].detach().cpu().numpy(),
                         re=rowEnds[si].detach().cpu().numpy())
                print(f"[dump] topk rows={sel[0]}..{sel[-1]} of {N} -> {path}",
                      file=_s.stderr, flush=True)
            return r
        return wrapper

    n = 0
    for mn, m in list(_s.modules.items()):
        if m is None or (mn != "aiter" and not mn.startswith("aiter.")):
            continue
        fn = getattr(m, "top_k_per_row_prefill", None)
        if fn is None or getattr(fn, "_tkdump", False):
            continue
        w = wrap(fn); w._tkdump = True
        setattr(m, "top_k_per_row_prefill", w); n += 1
    print(f"[dump] topk wrapper on {n} slot(s)", file=_s.stderr, flush=True)


def _patch_aiter(aiter_mod):
    if not _AUTO_TRITON:
        return
    try:
        index = _build_triton_index()
    except Exception as exc:
        _emit({"event": "triton_index_failed", "error": repr(exc)})
        return

    redirected = []
    replacements = {}

    for name, obj in list(vars(aiter_mod).items()):
        if name.startswith("_") or not callable(obj) or name in _DENY:
            continue
        origin = getattr(obj, "__module__", "") or ""
        # torch_compile_guard rewrites __module__ to aiter.jit.utils.torch_guard,
        # so match on the whole aiter namespace rather than just aiter.ops.
        if not origin.startswith("aiter.") or origin.startswith("aiter.ops.triton"):
            continue
        hit = index.get(name)
        if hit is None:
            continue
        replacements[name] = hit[0]
        redirected.append({"op": name, "from": origin, "to": hit[1]})

    import importlib

    for op, (tmod_name, tfunc_name) in _ALIASES.items():
        try:
            tfunc = getattr(importlib.import_module(tmod_name), tfunc_name)
        except Exception as exc:
            _emit({"event": "alias_failed", "op": op, "error": repr(exc)})
            continue
        replacements[op] = tfunc
        redirected.append({"op": op, "from": "alias", "to": f"{tmod_name}.{tfunc_name}"})

    try:
        for op, impl in _make_custom_impls().items():
            replacements[op] = impl
            redirected.append({"op": op, "from": "custom", "to": "shim adapter"})
    except Exception as exc:
        _emit({"event": "custom_impls_failed", "error": repr(exc)})

    if _COV:
        replacements = {op: _cov_wrap(op, impl) for op, impl in replacements.items()}
        _COV_ORIGIN.update({r["op"]: r["from"] for r in redirected})

    # Rebind across every already-imported aiter module, not just the package
    # top level: aiter's own internals call these by module-global name (e.g.
    # aiter/ops/quant.py::per_group_quant_hip -> dynamic_per_token_scaled_quant),
    # so patching only `aiter.X` leaves those call sites on the dead CK op.
    rebound = 0
    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        if mod_name != "aiter" and not mod_name.startswith("aiter."):
            continue
        if mod_name.startswith("aiter.ops.triton"):
            continue
        for op, impl in replacements.items():
            cur = getattr(mod, op, None)
            if cur is None or cur is impl or not callable(cur):
                continue
            try:
                setattr(mod, op, impl)
                rebound += 1
            except Exception:
                pass
    _emit({"event": "rebound_module_globals", "count": rebound})

    _tk = os.environ.get("GLM_DUMP_TOPK")
    if _tk:
        try:
            _install_topk_dump(_tk)
        except Exception as exc:
            _emit({"event": "topk_dump_failed", "error": repr(exc)})

    _emit({
        "event": "triton_redirect",
        "count": len(redirected),
        "ops": redirected,
    })


# --------------------------------------------------------------------------
# 2. CK/HIP build-failure reporting
# --------------------------------------------------------------------------

def _patch_compile_ops(core_mod):
    import functools

    orig_compile_ops = core_mod.compile_ops

    def compile_ops(_md_name, fc_name=None, *args, **kwargs):
        real_decorator = orig_compile_ops(_md_name, fc_name, *args, **kwargs)

        def decorator(func):
            built = real_decorator(func)
            op_name = fc_name if fc_name is not None else func.__name__
            state = {"dead": False}

            @functools.wraps(func)
            def guarded(*a, **kw):
                if state["dead"]:
                    raise Gfx1250MissingKernel(
                        f"aiter op {op_name!r} (module {_md_name!r}) has no "
                        f"gfx1250 build and no Triton peer"
                    )
                try:
                    return built(*a, **kw)
                except Exception as exc:
                    state["dead"] = True
                    _emit({
                        "event": "ck_hip_build_or_call_failed",
                        "op": op_name,
                        "aiter_module": _md_name,
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:400],
                    })
                    raise Gfx1250MissingKernel(
                        f"aiter op {op_name!r} (module {_md_name!r}) failed: "
                        f"{type(exc).__name__}: {str(exc)[:200]}"
                    ) from exc

            return guarded

        return decorator

    core_mod.compile_ops = compile_ops
    _emit({"event": "compile_ops_patched"})


# --------------------------------------------------------------------------
# 3. module-level breadcrumbs (GFX1250_TRACE=1)
# --------------------------------------------------------------------------

# A GPU page fault aborts the process inside the HSA queue with no Python
# frame, so the only way to localise it is to leave a trail: print every
# nn.Module entry/exit and read the last unbalanced one. Pair with
# HIP_LAUNCH_BLOCKING=1 so the fault lands in the module that caused it.
def _install_module_tracer(torch_mod):
    depth = [0]
    # Container modules dominate the output and never launch kernels.
    skip = {"ModuleList", "ModuleDict", "Sequential"}

    def pre(module, args):
        name = type(module).__name__
        if name in skip:
            return
        print(f"[trace]{'  ' * min(depth[0], 12)}> {name}", file=sys.stderr, flush=True)
        depth[0] += 1

    def post(module, args, output):
        name = type(module).__name__
        if name in skip:
            return
        depth[0] = max(0, depth[0] - 1)
        print(f"[trace]{'  ' * min(depth[0], 12)}< {name}", file=sys.stderr, flush=True)

    torch_mod.nn.modules.module.register_module_forward_pre_hook(pre)
    torch_mod.nn.modules.module.register_module_forward_hook(post)
    _emit({"event": "module_tracer_installed"})


def _install_logit_dumper(torch_mod, path):
    torch = torch_mod
    """Save the LM-head logits of the first forward, for cross-machine diffing."""
    # Overwrite on every call: the first LM-head forward is ATOM's startup
    # memory-profiling run on dummy inputs (which can be all-NaN), so the file
    # must end up holding the LAST forward -- the real request.
    def hook(module, args, output):
        if type(module).__name__ != "ParallelLMHead":
            return
        out = output[0] if isinstance(output, (tuple, list)) else output
        if not hasattr(out, "detach"):
            return
        # .cpu() is a device sync, which is illegal mid-capture (ATOM captures
        # decode graphs even under --cudagraph-mode NONE on some arches).
        try:
            if torch.cuda.is_current_stream_capturing():
                return
        except Exception:
            pass
        import numpy as np

        arr = out.detach().float().cpu().numpy()
        np.save(path, arr)
        print(f"[dump] logits {arr.shape} -> {path}", file=sys.stderr, flush=True)

    torch_mod.nn.modules.module.register_module_forward_hook(hook)


def _patch_torch(torch_mod):
    dump = os.environ.get("GLM_DUMP_LOGITS")
    if dump:
        _install_logit_dumper(torch_mod, dump)
    if os.environ.get("GFX1250_TRACE") != "1":
        return
    _install_module_tracer(torch_mod)


def _patch_triton(triton_mod):
    """Print every Triton kernel name at launch (GFX1250_TRACE_KERNELS=1)."""
    if os.environ.get("GFX1250_TRACE_KERNELS") != "1":
        return
    from triton.runtime.jit import JITFunction

    orig_run = JITFunction.run

    def run(self, *args, **kwargs):
        name = getattr(self, "__name__", None) or getattr(self.fn, "__name__", "?")
        print(f"[kernel] {name}", file=sys.stderr, flush=True)
        return orig_run(self, *args, **kwargs)

    JITFunction.run = run
    _emit({"event": "triton_kernel_tracer_installed"})


# --------------------------------------------------------------------------
# import hook
# --------------------------------------------------------------------------

_HOOKS = {
    "aiter.jit.core": _patch_compile_ops,
    "aiter": _patch_aiter,
    "torch": _patch_torch,
    "triton": _patch_triton,
}


class _PostExecPatcher:
    def find_spec(self, fullname, path=None, target=None):
        hook = _HOOKS.get(fullname)
        if hook is None:
            return None
        import importlib.util

        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        except Exception:
            spec = None
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None

        orig_exec = spec.loader.exec_module

        def exec_module(module, _orig=orig_exec, _hook=hook, _name=fullname):
            _orig(module)
            try:
                _hook(module)
            except Exception as exc:
                _emit({"event": "hook_failed", "module": _name, "error": repr(exc)})

        try:
            spec.loader.exec_module = exec_module
        except Exception:
            return None
        return spec


if _ENABLED and "aiter" not in sys.modules:
    sys.meta_path.insert(0, _PostExecPatcher())


# --------------------------------------------------------------------------
# Per-layer hidden-state tap, ported from the gfx950 reference shim so the two
# machines can be compared tap for tap.  GLM_ATTN0=<path> [GLM_N_LAYERS,
# GLM_FULL_CHUNK, GLM_TAP_STRIDE].  Composed onto the existing "torch" hook
# rather than adding a second meta_path finder.
# --------------------------------------------------------------------------
_TAP = os.environ.get("GLM_ATTN0")


def _install_attn0(torch_mod, path):
    torch = torch_mod
    TAGS = {"DeepseekV2MLAAttention": "attn", "DeepseekV2MoE": "ffn",
            "DeepseekV2MLP": "ffn", "DeepseekV2DecoderLayer": "layer"}
    NL = int(os.environ.get("GLM_N_LAYERS", "78"))
    STRIDE = int(os.environ.get("GLM_TAP_STRIDE", "1"))
    _w = os.environ.get("GLM_FULL_CHUNK", "1")
    WANT = None if _w == "all" else int(_w)
    st = {"L": 0, "chunk": 0}

    def hook(module, args, output):
        tag = TAGS.get(type(module).__name__)
        if tag is None:
            return
        try:
            from atom.utils.forward_context import get_forward_context
            if get_forward_context().context.is_dummy_run:
                return
        except Exception:
            pass
        try:
            if torch.cuda.is_current_stream_capturing():
                return
        except Exception:
            pass
        x = output[0] if isinstance(output, (tuple, list)) else output
        if not hasattr(x, "detach") or x.dim() < 2:
            return
        if x.shape[0] <= 2048:          # skip warmup/decode forwards
            return
        if tag == "layer":
            # The DecoderLayer output IS the residual stream; attn/ffn taps are
            # branch outputs and exaggerate disagreement.
            if (WANT is None or st["chunk"] == WANT) and torch.cuda.current_device() == 0:
                import numpy as _np
                _i = list(range(0, x.shape[0], STRIDE))
                if _i[-1] != x.shape[0] - 1:
                    _i.append(x.shape[0] - 1)
                _sfx = "" if WANT is not None else "c%d." % st["chunk"]
                _np.save("%s.%sres%d.npy" % (path, _sfx, st["L"]),
                         x[_i].detach().float().cpu().numpy())
                # DecoderLayer returns (hidden_states, residual); the SECOND
                # element is the accumulated residual stream.
                if isinstance(output, (tuple, list)) and len(output) > 1 and \
                        hasattr(output[1], "detach"):
                    _np.save("%s.%sstream%d.npy" % (path, _sfx, st["L"]),
                             output[1][_i].detach().float().cpu().numpy())
            st["L"] += 1
            if st["L"] >= NL:
                st["L"] = 0
                st["chunk"] += 1
            return
        if (WANT is not None and st["chunk"] != WANT) or torch.cuda.current_device() != 0:
            return
        import numpy as np
        # Subsample rows, but always keep the LAST one: that row is the token
        # whose hidden state decides the next generated token.
        _idx = list(range(0, x.shape[0], STRIDE))
        if _idx[-1] != x.shape[0] - 1:
            _idx.append(x.shape[0] - 1)
        arr = x[_idx].detach().float().cpu().numpy()
        sfx = "" if WANT is not None else "c%d." % st["chunk"]
        np.save("%s.%s%s%d.npy" % (path, sfx, tag, st["L"]), arr)
        print("[tap] chunk%d %s%d %s" % (st["chunk"], tag, st["L"], arr.shape),
              file=sys.stderr, flush=True)

    torch_mod.nn.modules.module.register_module_forward_hook(hook)
    print("[tap] installed (stride=%d)" % STRIDE, file=sys.stderr, flush=True)


if _TAP:
    _prev_torch_hook = _HOOKS.get("torch")

    def _patch_torch_with_tap(mod, _p=_prev_torch_hook):
        if _p is not None:
            _p(mod)
        try:
            _install_attn0(mod, _TAP)
        except Exception as exc:
            print("[tap] failed: %r" % (exc,), file=sys.stderr, flush=True)

    _HOOKS["torch"] = _patch_torch_with_tap

# --------------------------------------------------------------------------
# Layer-0 bisect dump, identical on gfx950 and gfx1250.  GLM_L0=<path>
# Captures, for the SECOND prefill chunk (numRows != the first chunk's size):
#   .emb.npy  embedding output (layer-0 input), every 16th token
#   .l0.npz   layer-0 indexer logits for 16 probe rows, plus the SORTED index
#             set and the row extents, so selection can be compared
#             order-insensitively across two different top-k implementations.
# --------------------------------------------------------------------------
_L0 = os.environ.get("GLM_L0")


def _install_l0(aiter_mod):
    import sys as _s
    import torch as _t
    st = {"first_rows": None, "done": False}

    def wrap(orig):
        def wrapper(logits, rowStarts, rowEnds, indices, values, numRows,
                    stride0, stride1, k=2048, *a, **kw):
            r = orig(logits, rowStarts, rowEnds, indices, values, numRows,
                     stride0, stride1, k, *a, **kw)
            try:
                n = int(numRows)
                if n <= 2048 or _t.cuda.current_device() != 0:
                    return r
                if st["first_rows"] is None:
                    st["first_rows"] = n          # chunk 1 size
                    return r
                if n == st["first_rows"] or st["done"]:
                    return r
                st["done"] = True                  # first chunk-2 call = layer 0
                import numpy as np
                probe = np.arange(0, n, max(1, n // 16))[:16]
                pt = _t.as_tensor(probe, device=logits.device, dtype=_t.long)
                idx = indices[:n][pt].detach().cpu().numpy()
                np.savez(f"{_L0}.l0.npz",
                         probe=probe,
                         lg=logits[:n][pt].detach().float().cpu().numpy(),
                         idx=np.sort(idx, axis=1),
                         idx_raw=idx,
                         rs=rowStarts[:n][pt].detach().cpu().numpy(),
                         re=rowEnds[:n][pt].detach().cpu().numpy(),
                         nrows=np.array([n]))
                print(f"[l0] dumped chunk2 layer0 topk rows={n} -> {_L0}.l0.npz",
                      file=_s.stderr, flush=True)
            except Exception as exc:
                print(f"[l0] failed: {exc!r}", file=_s.stderr, flush=True)
            return r
        return wrapper

    cnt = 0
    for mn, m in list(_s.modules.items()):
        if m is None or (mn != "aiter" and not mn.startswith("aiter.")):
            continue
        fn = getattr(m, "top_k_per_row_prefill", None)
        if fn is None or getattr(fn, "_l0", False):
            continue
        w = wrap(fn); w._l0 = True
        setattr(m, "top_k_per_row_prefill", w); cnt += 1
    print(f"[l0] topk wrapper on {cnt} slot(s)", file=_s.stderr, flush=True)


def _install_l0_emb(torch_mod):
    torch = torch_mod
    seen = {"first": None, "done": False}

    def hook(module, args, output):
        if "Embedding" not in type(module).__name__:
            return
        x = output[0] if isinstance(output, (tuple, list)) else output
        if not hasattr(x, "detach") or x.dim() != 2 or x.shape[0] <= 2048:
            return
        if seen["first"] is None:
            seen["first"] = x.shape[0]
            return
        if x.shape[0] == seen["first"] or seen["done"]:
            return
        if torch.cuda.current_device() != 0:
            return
        seen["done"] = True
        import numpy as np
        np.save(f"{_L0}.emb.npy", x[::16].detach().float().cpu().numpy())
        print(f"[l0] dumped embeddings {tuple(x.shape)}", file=sys.stderr, flush=True)

    torch_mod.nn.modules.module.register_module_forward_hook(hook)


if _L0:
    _prev_aiter = _HOOKS.get("aiter")
    _prev_torch2 = _HOOKS.get("torch")

    def _aiter_with_l0(mod, _p=_prev_aiter):
        if _p is not None:
            _p(mod)
        try:
            _install_l0(mod)
        except Exception as exc:
            print("[l0] topk hook failed: %r" % (exc,), file=sys.stderr, flush=True)

    def _torch_with_l0(mod, _p=_prev_torch2):
        if _p is not None:
            _p(mod)
        try:
            _install_l0_emb(mod)
        except Exception as exc:
            print("[l0] emb hook failed: %r" % (exc,), file=sys.stderr, flush=True)

    _HOOKS["aiter"] = _aiter_with_l0
    _HOOKS["torch"] = _torch_with_l0

# --------------------------------------------------------------------------
# Capture the ACTUAL inputs to the sparse-MLA prefill at layer 0 of the second
# chunk, identically on both machines, so the op can be replayed on CPU.
# GLM_MLAIN=<path>
# --------------------------------------------------------------------------
_MLAIN = os.environ.get("GLM_MLAIN")


def _install_mlain(aiter_mod):
    import sys as _s
    import torch as _t
    st = {"first": None, "done": False}

    def wrap(orig):
        def wrapper(q, kv_buffer, qo_indptr, kv_indptr, kv_indices,
                    kv_last_page_lens, max_seqlen_q, sm_scale, logits,
                    attn_lse, *a, **kw):
            try:
                n = int(qo_indptr.shape[0]) - 1
                if st["first"] is None:
                    st["first"] = n
                elif n != st["first"] and not st["done"]:
                    st["done"] = True
                    import numpy as np
                    probe = np.arange(0, n, max(1, n // 4))[:4]
                    kvp = kv_indptr.detach().cpu().numpy().astype(np.int64)
                    kvi = kv_indices.detach().cpu().numpy().astype(np.int64)
                    qo = qo_indptr.detach().cpu().numpy().astype(np.int64)
                    sel = {}
                    for r in probe:
                        sel[int(r)] = kvi[kvp[r]:kvp[r + 1]]
                    union = np.unique(np.concatenate(list(sel.values())))
                    npg, psz, nkv, hd = kv_buffer.shape
                    flat = kv_buffer.reshape(npg * psz, hd)
                    ut = _t.as_tensor(union, device=kv_buffer.device, dtype=_t.long)
                    qrows = np.concatenate([np.arange(qo[r], qo[r + 1]) for r in probe])
                    qt = _t.as_tensor(qrows, device=q.device, dtype=_t.long)
                    np.savez(f"{_MLAIN}.mlain.npz",
                             probe=probe, ngroups=np.array([n]),
                             q=q[qt].detach().float().cpu().numpy(),
                             qrows=qrows,
                             union=union,
                             kv=flat[ut].detach().float().cpu().numpy(),
                             kv_shape=np.array(kv_buffer.shape),
                             kv_dtype=np.array([str(kv_buffer.dtype)]),
                             sel=np.array([sel[int(r)] for r in probe]),
                             sm_scale=np.array([float(sm_scale)]),
                             max_seqlen_q=np.array([int(max_seqlen_q)]),
                             v_head_dim=np.array([int(logits.shape[-1])]),
                             qshape=np.array(q.shape))
                    st["qt"] = qt
                    print(f"[mlain] dumped: q{tuple(q.shape)} kv{tuple(kv_buffer.shape)} "
                          f"union={union.size} -> {_MLAIN}.mlain.npz",
                          file=_s.stderr, flush=True)
            except Exception as exc:
                print(f"[mlain] failed: {exc!r}", file=_s.stderr, flush=True)
            r = orig(q, kv_buffer, qo_indptr, kv_indptr, kv_indices,
                     kv_last_page_lens, max_seqlen_q, sm_scale, logits,
                     attn_lse, *a, **kw)
            _qt = st.pop('qt', None)
            if _qt is not None:
                try:
                    import numpy as np
                    o = logits.reshape(-1, logits.shape[-2],
                                       logits.shape[-1])[_qt]
                    np.save(f'{_MLAIN}.acore.npy',
                            o.detach().float().cpu().numpy())
                    print(f'[mlain] acore {tuple(o.shape)}',
                          file=_s.stderr, flush=True)
                except Exception as exc:
                    print(f'[mlain] acore failed: {exc!r}',
                          file=_s.stderr, flush=True)
            return r
        return wrapper

    cnt = 0
    for mn, m in list(_s.modules.items()):
        if m is None or (mn != "aiter" and not mn.startswith("aiter.")):
            continue
        fn = getattr(m, "mla_prefill_asm_fwd", None)
        if fn is None or getattr(fn, "_mlain", False):
            continue
        w = wrap(fn); w._mlain = True
        setattr(m, "mla_prefill_asm_fwd", w); cnt += 1
    print(f"[mlain] wrapper on {cnt} slot(s)", file=_s.stderr, flush=True)


if _MLAIN:
    _prev_aiter_m = _HOOKS.get("aiter")

    def _aiter_with_mlain(mod, _p=_prev_aiter_m):
        if _p is not None:
            _p(mod)
        try:
            _install_mlain(mod)
        except Exception as exc:
            print("[mlain] hook failed: %r" % (exc,), file=sys.stderr, flush=True)

    _HOOKS["aiter"] = _aiter_with_mlain

# --------------------------------------------------------------------------
# Capture the inputs/outputs of fused_qk_rope_concat_and_cache_mla at layer 0
# of the second prefill chunk, so a replacement can be developed and validated
# offline.  GLM_FQK=<path>
# --------------------------------------------------------------------------
_FQK = os.environ.get("GLM_FQK")


def _install_fqk(aiter_mod):
    import sys as _s
    import torch as _t
    st = {"first": None, "done": False}

    def wrap(orig):
        def wrapper(q_nope, q_pe, kv_c, k_pe, kv_cache, q_out, slot_mapping,
                    k_scale, q_scale, positions, cos_cache, sin_cache,
                    is_neox, is_nope_first, *a, **kw):
            r = orig(q_nope, q_pe, kv_c, k_pe, kv_cache, q_out, slot_mapping,
                     k_scale, q_scale, positions, cos_cache, sin_cache,
                     is_neox, is_nope_first, *a, **kw)
            try:
                n = int(q_pe.shape[0])
                if st["first"] is None:
                    st["first"] = n
                elif n != st["first"] and not st["done"]:
                    st["done"] = True
                    import numpy as np
                    P = np.arange(0, n, max(1, n // 4))[:4]
                    pt = _t.as_tensor(P, device=q_pe.device, dtype=_t.long)
                    sm = slot_mapping.detach().cpu().numpy().astype(np.int64)
                    bs = int(kv_cache.shape[1])
                    slots = sm[P]
                    bi = _t.as_tensor(slots // bs, device=kv_cache.device, dtype=_t.long)
                    bo = _t.as_tensor(slots % bs, device=kv_cache.device, dtype=_t.long)
                    np.savez(f"{_FQK}.fqk.npz",
                             probe=P, slots=slots,
                             q_nope=q_nope[pt].detach().float().cpu().numpy(),
                             q_pe=q_pe[pt].detach().float().cpu().numpy(),
                             q_out=q_out[pt].detach().float().cpu().numpy(),
                             kv_c=kv_c[pt].detach().float().cpu().numpy(),
                             k_pe=k_pe[pt].detach().float().cpu().numpy(),
                             cache_after=kv_cache[bi, bo].detach().float().cpu().numpy(),
                             positions=positions.detach().cpu().numpy()[P],
                             cos=cos_cache.detach().float().cpu().numpy(),
                             sin=sin_cache.detach().float().cpu().numpy(),
                             is_neox=np.array([bool(is_neox)]),
                             is_nope_first=np.array([bool(is_nope_first)]),
                             k_scale=k_scale.detach().float().cpu().numpy().ravel()[:4],
                             q_scale=q_scale.detach().float().cpu().numpy().ravel()[:4],
                             kv_cache_shape=np.array(kv_cache.shape),
                             kv_cache_dtype=np.array([str(kv_cache.dtype)]),
                             q_out_shape=np.array(q_out.shape),
                             ntok=np.array([n]))
                    print(f"[fqk] dumped n={n} neox={bool(is_neox)} "
                          f"nope_first={bool(is_nope_first)} kv{tuple(kv_cache.shape)} "
                          f"qout{tuple(q_out.shape)}", file=_s.stderr, flush=True)
            except Exception as exc:
                print(f"[fqk] failed: {exc!r}", file=_s.stderr, flush=True)
            return r
        return wrapper

    cnt = 0
    for mn, m in list(_s.modules.items()):
        if m is None or (mn != "aiter" and not mn.startswith("aiter.")):
            continue
        fn = getattr(m, "fused_qk_rope_concat_and_cache_mla", None)
        if fn is None or getattr(fn, "_fqk", False):
            continue
        w = wrap(fn); w._fqk = True
        setattr(m, "fused_qk_rope_concat_and_cache_mla", w); cnt += 1
    print(f"[fqk] wrapper on {cnt} slot(s)", file=_s.stderr, flush=True)


if _FQK:
    _prev_a = _HOOKS.get("aiter")

    def _aiter_with_fqk(mod, _p=_prev_a):
        if _p is not None:
            _p(mod)
        try:
            _install_fqk(mod)
        except Exception as exc:
            print("[fqk] hook failed: %r" % (exc,), file=sys.stderr, flush=True)

    _HOOKS["aiter"] = _aiter_with_fqk

# --------------------------------------------------------------------------
# Which op actually writes the MLA KV cache?  GLM_PROBE_KV=1
# ATOM binds these by value and wraps them in mark_trace, so patch them in
# atom.model_ops.attention_mla's own namespace, not aiter's.
# --------------------------------------------------------------------------
_PROBEKV = os.environ.get("GLM_PROBE_KV")

_KV_WRITERS = [
    "concat_and_cache_mla", "concat_and_cache_mla_seg",
    "fused_qk_rope_concat_and_cache_mla", "fused_qk_rope_concat_and_cache_mla_seg",
    "triton_cat_and_cache_mla", "triton_fused_qk_rope_cat_and_cache_mla",
]


def _install_probe_kv(mod):
    import sys as _s
    counts = {}

    def mk(name, fn):
        def w(*a, **kw):
            counts[name] = counts.get(name, 0) + 1
            if counts[name] in (1, 80, 157):
                shapes = [tuple(x.shape) for x in a if hasattr(x, "shape")][:6]
                print(f"[kvprobe] {name} call#{counts[name]} shapes={shapes}",
                      file=_s.stderr, flush=True)
            return fn(*a, **kw)
        w._probed = True
        return w

    found = []
    for nm in _KV_WRITERS:
        fn = getattr(mod, nm, None)
        if fn is None or getattr(fn, "_probed", False):
            continue
        setattr(mod, nm, mk(nm, fn)); found.append(nm)
    print(f"[kvprobe] wrapped in {mod.__name__}: {found}", file=_s.stderr, flush=True)


if _PROBEKV:
    _HOOKS["atom.model_ops.attention_mla"] = _install_probe_kv

# --------------------------------------------------------------------------
# gfx1250: route the HIP fused_qk_rope_concat_and_cache_mla (module_cache does
# not build here) to aiter's TRITON fused_qk_rope_cat_and_cache_mla with
# shuffled_kv_cache=False, so the whole stack stays in the UNSHUFFLED layout
# that the sparse reader (unified_attention_sparse_mla / the torch
# mla_prefill_asm_fwd reference) is the only layout able to consume.
# Enable with ATOM_USE_TRITON_MLA_SHUFFLE_KV=0 and GLM_FQK_TRITON=1.
# --------------------------------------------------------------------------
_FQKFIX = os.environ.get("GLM_FQK_TRITON")


def _install_fqk_triton(mod):
    import sys as _s
    import torch as _t
    from aiter.ops.triton.fusions.fused_kv_cache import (
        fused_qk_rope_cat_and_cache_mla as _tri,
    )
    seen = {"n": 0}

    def repl(q_nope, q_pe, kv_c, k_pe, kv_cache, q_out, slot_mapping,
             k_scale, q_scale, positions, cos_cache, sin_cache,
             is_neox, is_nope_first, *a, **kw):
        dim = kv_cache.shape[-1]
        # Unshuffled convention: [num_slots, num_kv_heads, dim], indexed by slot.
        kvh = kv_c.shape[1] if kv_c.dim() == 3 else 1
        cache = kv_cache.reshape(-1, 1, dim)
        kn = kv_c if kv_c.dim() == 3 else kv_c.view(-1, kvh, kv_c.shape[-1])
        kp = k_pe if k_pe.dim() == 3 else k_pe.view(-1, kvh, k_pe.shape[-1])
        seen["n"] += 1
        if seen["n"] in (1, 2, 157):
            print(f"[fqkfix] call#{seen['n']} q_nope{tuple(q_nope.shape)} "
                  f"kn{tuple(kn.shape)} cache{tuple(cache.shape)} "
                  f"nope_first={bool(is_nope_first)} neox={bool(is_neox)}",
                  file=_s.stderr, flush=True)
        _tri(q_nope, q_pe, kn, kp, cache, slot_mapping, positions,
             cos_cache, sin_cache, k_scale, is_neox,
             num_decode_toks_for_zeros=0, apply_scale=True, q_out=q_out,
             shuffled_kv_cache=False)

    repl._fqkfix = True
    n = 0
    for nm in ("fused_qk_rope_concat_and_cache_mla",):
        if getattr(getattr(mod, nm, None), "_fqkfix", False):
            continue
        if hasattr(mod, nm):
            setattr(mod, nm, repl); n += 1
    print(f"[fqkfix] redirected {n} binding(s) in {mod.__name__} to the triton writer",
          file=_s.stderr, flush=True)


if _FQKFIX:
    _prev_am = _HOOKS.get("atom.model_ops.attention_mla")

    def _am_with_fqkfix(mod, _p=_prev_am):
        if _p is not None:
            _p(mod)
        try:
            _install_fqk_triton(mod)
        except Exception as exc:
            print("[fqkfix] failed: %r" % (exc,), file=sys.stderr, flush=True)

    _HOOKS["atom.model_ops.attention_mla"] = _am_with_fqkfix

# --------------------------------------------------------------------------
# Dump the Q halves entering the fused rope+cache op, to separate the q
# projection path from rope.  GLM_QDUMP=<path>
# Wrapped in atom.model_ops.attention_mla's namespace (ATOM binds by value).
# --------------------------------------------------------------------------
_QD = os.environ.get("GLM_QDUMP")


def _install_qdump(mod):
    import sys as _s
    import torch as _t
    st = {"first": None, "done": False}
    nm = "fused_qk_rope_concat_and_cache_mla"
    orig = getattr(mod, nm, None)
    if orig is None or getattr(orig, "_qd", False):
        return

    def w(q_nope, q_pe, kv_c, k_pe, kv_cache, q_out, slot_mapping,
          k_scale, q_scale, positions, cos_cache, sin_cache,
          is_neox, is_nope_first, *a, **kw):
        r = orig(q_nope, q_pe, kv_c, k_pe, kv_cache, q_out, slot_mapping,
                 k_scale, q_scale, positions, cos_cache, sin_cache,
                 is_neox, is_nope_first, *a, **kw)
        try:
            n = int(q_pe.shape[0])
            if n <= 2048:
                return r
            if st["first"] is None:
                st["first"] = n
            elif n != st["first"] and not st["done"]:
                st["done"] = True
                import numpy as np
                P = np.arange(0, n, max(1, n // 4))[:4]
                pt = _t.as_tensor(P, device=q_pe.device, dtype=_t.long)
                np.savez(f"{_QD}.q.npz", probe=P,
                         q_nope=q_nope[pt].detach().float().cpu().numpy(),
                         q_pe=q_pe[pt].detach().float().cpu().numpy(),
                         q_out=q_out[pt].detach().float().cpu().numpy(),
                         positions=positions.detach().cpu().numpy()[P],
                         cos=cos_cache.detach().float().cpu().numpy(),
                         sin=sin_cache.detach().float().cpu().numpy(),
                         is_neox=np.array([bool(is_neox)]),
                         is_nope_first=np.array([bool(is_nope_first)]))
                print(f"[qd] dumped n={n} q_nope{tuple(q_nope.shape)} "
                      f"q_pe{tuple(q_pe.shape)} q_out{tuple(q_out.shape)}",
                      file=_s.stderr, flush=True)
        except Exception as exc:
            print(f"[qd] failed: {exc!r}", file=_s.stderr, flush=True)
        return r

    w._qd = True
    setattr(mod, nm, w)
    print(f"[qd] wrapped {nm} in {mod.__name__}", file=_s.stderr, flush=True)


if _QD:
    _prev_qd = _HOOKS.get("atom.model_ops.attention_mla")

    def _am_with_qd(mod, _p=_prev_qd):
        if _p is not None:
            _p(mod)
        try:
            _install_qdump(mod)
        except Exception as exc:
            print("[qd] hook failed: %r" % (exc,), file=sys.stderr, flush=True)

    _HOOKS["atom.model_ops.attention_mla"] = _am_with_qd

# --------------------------------------------------------------------------
# Call counter for specific kernels: answers "is this op actually on the
# execution path?".  GLM_CALLCNT=<path>.  Sweeps every loaded module for the
# target attribute names and wraps them, so by-value bindings (ATOM copies
# aiter functions into its own namespace) are counted where they are called.
# The sweep runs from a module forward pre-hook, repeated a few times so late
# imports are still caught.
# --------------------------------------------------------------------------
_CALLCNT = os.environ.get("GLM_CALLCNT")

_CNT_TARGETS = (
    # 1 DSA indexer logits
    "fp8_mqa_logits",
    # 2 MLA decode / prefill attention
    "mla_decode_fwd", "mla_prefill_fwd", "mla_prefill_asm_fwd",
    "decode_attention_fwd", "triton_shuffle_mla_decode_fwd",
    "unified_attention_sparse_mla",
    # 3 preshuffled vs plain MXFP4 GEMM
    "gemm_afp4wfp4_preshuffle", "gemm_afp4wfp4", "gemm_a4w4", "gemm_a4w4_quant",
    # 4 MoE a4w4 / scale shuffle
    "shuffle_scale_moe", "moe_gemm_a4w4", "moe_gemm_a16w4", "fused_moe",
    # 5 fused KV cache writer
    "fused_qk_rope_cat_and_cache_mla", "fused_qk_rope_concat_and_cache_mla",
    # 6 skinny GEMM
    "is_skinny_default_shape", "wv_splitk_small_fp16_bf16", "wvSpltK", "LLMM1",
    # 7 the two absorption BMMs
    "batched_gemm_a16wfp4",
    "batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant",
)


def _install_callcnt(torch_mod):
    import atexit
    import functools
    counts = {}
    state = {"n": 0, "swept": 0}

    def wrap(mod_name, name, fn):
        key = f"{name}  @{mod_name}"
        counts.setdefault(key, 0)

        @functools.wraps(fn)
        def w(*a, **kw):
            counts[key] += 1
            return fn(*a, **kw)
        w._glm_cnt = True
        return w

    def sweep():
        state["swept"] += 1
        for mn, m in list(sys.modules.items()):
            if m is None:
                continue
            if not (mn == "aiter" or mn.startswith("aiter.")
                    or mn == "atom" or mn.startswith("atom.")):
                continue
            for name in _CNT_TARGETS:
                fn = getattr(m, name, None)
                if fn is None or not callable(fn) or getattr(fn, "_glm_cnt", False):
                    continue
                try:
                    setattr(m, name, wrap(mn, name, fn))
                except Exception:
                    pass

    def pre(module, args):
        state["n"] += 1
        if state["n"] in (1, 20, 200, 2000):
            sweep()

    torch_mod.nn.modules.module.register_module_forward_pre_hook(pre)

    def report():
        try:
            lines = ["called  op  @module"]
            for k in sorted(counts, key=lambda x: (-counts[x], x)):
                lines.append(f"{counts[k]:>7}  {k}")
            txt = "\n".join(lines) + f"\n(sweeps: {state['swept']})\n"
            p = "%s.%d" % (_CALLCNT, os.getpid())
            open(p, "w").write(txt)
            print("[cnt] wrote " + p, file=sys.stderr, flush=True)
        except Exception as exc:
            print(f"[cnt] report failed: {exc!r}", file=sys.stderr, flush=True)

    atexit.register(report)
    print("[cnt] installed", file=sys.stderr, flush=True)


if _CALLCNT:
    _prev_torch_cnt = _HOOKS.get("torch")

    def _torch_with_cnt(mod, _p=_prev_torch_cnt):
        if _p is not None:
            _p(mod)
        try:
            _install_callcnt(mod)
        except Exception as exc:
            print("[cnt] hook failed: %r" % (exc,), file=sys.stderr, flush=True)

    _HOOKS["torch"] = _torch_with_cnt

# --------------------------------------------------------------------------
# Replace the shim's torch sparse-MLA reference with aiter's Triton
# unified_attention_sparse_mla.  GLM_TRITON_SPARSE_MLA=1
# Ported from the gfx950 shim; valid only for the UNSHUFFLED cache layout,
# i.e. ATOM_USE_TRITON_MLA_SHUFFLE_KV=0 + GLM_FQK_TRITON=1.
# --------------------------------------------------------------------------
_TSM = os.environ.get("GLM_TRITON_SPARSE_MLA")
_TSM_LSE_POISON = os.environ.get("GLM_TSM_LSE_POISON") == "1"
_TSM_TRACE = os.environ.get("GLM_TSM_TRACE") == "1"


def _install_triton_sparse_mla(aiter_mod):
    import sys as _s
    import torch
    from aiter.ops.triton.attention.unified_attention_sparse_mla import (
        unified_attention_sparse_mla,
    )
    calls = {"n": 0}

    def mla_prefill_asm_fwd(q, kv_buffer, qo_indptr, kv_indptr, kv_indices,
                            kv_last_page_lens, max_seqlen_q, sm_scale,
                            logits, attn_lse, *a, **kw):
        calls["n"] += 1
        if _TSM_TRACE:
            print(
                "[tsmtrace] call#%d ENTER q=%s kv_buf=%s kv_indptr=%d "
                "qo_indptr=%d kv_indices=%d maxq=%s logits=%s"
                % (calls["n"], tuple(q.shape), tuple(kv_buffer.shape),
                   kv_indptr.numel(), qo_indptr.numel(), kv_indices.numel(),
                   max_seqlen_q, tuple(logits.shape)),
                file=_s.stderr, flush=True,
            )
        if calls["n"] in (1, 8, 40):
            print(f"[tsm] triton sparse-MLA call #{calls['n']} q={tuple(q.shape)}",
                  file=_s.stderr, flush=True)
        d = kv_buffer.shape[-1]
        flat = kv_buffer.reshape(-1, d)
        n = flat.shape[0]
        # The kernel derives TILE_SIZE from kv.shape[1]; ATOM hands a page_size=1
        # view, which would give a 1-wide tile. Re-viewing the same storage as
        # 64-slot blocks is a pure reshape: slot s -> (s // 64, s % 64) is exactly
        # the addressing the kernel does.
        tile = 64 if n % 64 == 0 else (16 if n % 16 == 0 else 1)
        kv4 = flat.view(-1, tile, 1, d)

        nheads, v_dim = logits.shape[-2], logits.shape[-1]
        out = logits.reshape(-1, nheads, v_dim)
        t = out.shape[0]

        # CSR -> dense [T, K] with -1 padding: the top-k form the kernel wants.
        kvp = kv_indptr[: t + 1].to(torch.int64)
        counts = (kvp[1:] - kvp[:-1]).clamp_min(0)
        k_max = int(counts.max().item())
        if _TSM_TRACE:
            _bad = int((kv_indices >= (kv_buffer.numel() // d)).sum().item())
            _neg = int((kv_indices < -1).sum().item())
            print(
                "[tsmtrace] call#%d MARSHAL t=%d k_max=%d n_slots=%d "
                "empty_rows=%d oob_idx=%d neg_idx=%d idx_max=%d"
                % (calls["n"], t, k_max, kv_buffer.numel() // d,
                   int((counts == 0).sum().item()), _bad, _neg,
                   int(kv_indices.max().item())),
                file=_s.stderr, flush=True,
            )
        if k_max == 0:
            out.zero_()
            if attn_lse is not None:
                attn_lse.zero_()
            return
        ar = torch.arange(k_max, device=q.device)
        gather = (kvp[:-1].unsqueeze(1) + ar.unsqueeze(0)).clamp(
            0, kv_indices.numel() - 1)
        dense = kv_indices.to(torch.int32)[gather]
        dense = torch.where(ar.unsqueeze(0) < counts.unsqueeze(1), dense,
                            torch.full_like(dense, -1)).contiguous()

        # block_table / seqused_k are dead args here (block_tables_ptr is never
        # dereferenced, seq_lens_ptr is unused); only cu_seqlens_q matters.
        unified_attention_sparse_mla(
            q, kv4, out,
            qo_indptr[: t + 1].to(torch.int32),
            int(max_seqlen_q),
            counts.to(torch.int32),
            k_max,
            sm_scale,
            dense,
            torch.zeros((t, 1), dtype=torch.int32, device=q.device),
            v_dim,
        )
        if attn_lse is not None:
            if _TSM_LSE_POISON:
                attn_lse.fill_(float("nan"))
            else:
                attn_lse.zero_()

    cnt = 0
    for mn, m in list(_s.modules.items()):
        if m is None or (mn != "aiter" and not mn.startswith("aiter.")):
            continue
        if getattr(m, "mla_prefill_asm_fwd", None) is not None:
            setattr(m, "mla_prefill_asm_fwd", mla_prefill_asm_fwd)
            cnt += 1
    print(f"[tsm] triton unified_attention_sparse_mla bound in {cnt} slot(s)",
          file=_s.stderr, flush=True)
    # The MLAIN capture wrapped the torch impl we just replaced; re-attach it.
    if _MLAIN:
        _install_mlain(aiter_mod)


if _TSM == "1":
    _prev_a_tsm = _HOOKS.get("aiter")

    def _aiter_with_tsm(mod, _p=_prev_a_tsm):
        if _p is not None:
            _p(mod)
        try:
            _install_triton_sparse_mla(mod)
        except Exception as exc:
            print("[tsm] hook failed: %r" % (exc,), file=sys.stderr, flush=True)

    _HOOKS["aiter"] = _aiter_with_tsm

# --------------------------------------------------------------------------
# Separate "many staging buffers are LIVE" from "freed buffers are RETAINED by
# the caching pinned-host allocator".  GLM_STAGE_STATS=<path>
# A weakref finalizer on each staging tensor decrements the live counters the
# moment the buffer is collected, so peak-live is measured independently of
# whatever the allocator does with the freed block.
# --------------------------------------------------------------------------
_STG = os.environ.get("GLM_STAGE_STATS")


def _install_stage_stats(mod):
    import atexit
    import sys as _s
    import threading
    import weakref

    import torch

    P = mod.ExpertStagingPool
    orig = P.__dict__["_allocate_staging"]
    orig_fn = orig.__func__ if isinstance(orig, staticmethod) else orig
    st = {"live": 0, "live_b": 0, "peak": 0, "peak_b": 0, "n": 0, "cum_b": 0,
          "pinned": 0, "unpinned": 0}
    lk = threading.Lock()

    def _dec(nb):
        with lk:
            st["live"] -= 1
            st["live_b"] -= nb

    def alloc(param):
        t = orig_fn(param)
        nb = t.numel() * t.element_size()
        with lk:
            st["live"] += 1
            st["live_b"] += nb
            st["n"] += 1
            st["cum_b"] += nb
            st["pinned" if t.is_pinned() else "unpinned"] += 1
            if st["live"] > st["peak"]:
                st["peak"] = st["live"]
            if st["live_b"] > st["peak_b"]:
                st["peak_b"] = st["live_b"]
        weakref.finalize(t, _dec, nb)
        return t

    P._allocate_staging = staticmethod(alloc)

    # Per-entry lifetime: which fused params stay open, and for how many other
    # allocations. Separates "the reader races ahead" from "one param's
    # arrivals are spread across the whole checkpoint".
    _stg_names = {}
    _lifetimes = []
    orig_goce = P._get_or_create_entry

    def goce(self, param, moe, name):
        before = st["n"]
        entry = orig_goce(self, param, moe, name)
        if entry is not None and id(entry) not in _stg_names:
            _stg_names[id(entry)] = name
            born = st["n"]

            def _died(_n=name, _b=born):
                with lk:
                    _lifetimes.append((_n, _b, st["n"]))
            weakref.finalize(entry.staging, _died)
        return entry

    P._get_or_create_entry = goce

    def host_stats():
        try:
            h = torch.cuda.host_memory_stats()
            return (h.get("allocated_bytes.all.current", 0),
                    h.get("reserved_bytes.all.current", 0))
        except Exception:
            return (0, 0)

    def sampler():
        G = 1 << 30
        while True:
            with lk:
                s = dict(st)
            ha, hr = host_stats()
            try:
                avail = int([l for l in open("/proc/meminfo")
                             if l.startswith("MemAvailable")][0].split()[1]) / (1 << 20)
            except Exception:
                avail = -1
            print("[stg] live=%d/%.1fGB peak=%d/%.1fGB allocs=%d cum=%.1fGB "
                  "host_alloc=%.1fGB host_reserved=%.1fGB avail=%.1fGB"
                  % (s["live"], s["live_b"] / G, s["peak"], s["peak_b"] / G,
                     s["n"], s["cum_b"] / G, ha / G, hr / G, avail),
                  file=_s.stderr, flush=True)
            threading.Event().wait(5)

    threading.Thread(target=sampler, daemon=True).start()

    def report():
        G = 1 << 30
        ha, hr = host_stats()
        line = ("[stg] FINAL peak_live_entries=%d peak_live_bytes=%.1fGB "
                "total_allocs=%d cumulative_bytes=%.1fGB pinned=%d unpinned=%d "
                "host_alloc=%.1fGB host_reserved=%.1fGB"
                % (st["peak"], st["peak_b"] / G, st["n"], st["cum_b"] / G,
                   st["pinned"], st["unpinned"], ha / G, hr / G))
        print(line, file=_s.stderr, flush=True)
        try:
            with open("%s.%d" % (_STG, os.getpid()), "w") as fh:
                fh.write(line + "\n")
                spans = sorted(_lifetimes, key=lambda x: x[2] - x[1], reverse=True)
                fh.write("longest-lived staging entries "
                         "(allocs that happened while it was open):\n")
                for nm, b, d in spans[:12]:
                    fh.write("  %6d allocs  %s\n" % (d - b, nm))
                fh.write("median span: %d allocs over %d entries\n"
                         % (sorted(d - b for _, b, d in spans)[len(spans) // 2]
                            if spans else -1, len(spans)))
        except Exception as exc:
            print("[stg] dump failed %r" % (exc,), file=_s.stderr, flush=True)

    atexit.register(report)
    print("[stg] staging accounting installed", file=_s.stderr, flush=True)


if _STG:
    _HOOKS["atom.model_loader.expert_staging"] = _install_stage_stats

# --------------------------------------------------------------------------
# Why does a staging entry stay open?  GLM_STAGE_EVENTS=1
# Replaces ExpertStagingPool._entries with a dict that logs every insert and
# every removal, attributing the removal to its caller (stage / decline /
# flush_pending) via the calling frame. That distinguishes "the entry finally
# completed" from "another loader path declined it" from "it was still open at
# the end of the load".
# --------------------------------------------------------------------------
_STGEV = os.environ.get("GLM_STAGE_EVENTS")


def _install_stage_events(mod):
    import sys as _s
    import threading

    P = mod.ExpertStagingPool
    seq = {"n": 0}
    lk = threading.Lock()

    class _LogDict(dict):
        def setdefault(self, k, v):
            was = k in self
            r = super().setdefault(k, v)
            if not was:
                with lk:
                    seq["n"] += 1
                    n = seq["n"]
                print("[stgev] OPEN  seq=%d live=%d expected=%d name=%s"
                      % (n, len(self), getattr(v, "expected", -1),
                         getattr(v, "name", "?")),
                      file=_s.stderr, flush=True)
            return r

        def pop(self, k, *a):
            v = super().pop(k, *a)
            if v is not None and hasattr(v, "name"):
                try:
                    caller = _s._getframe(1).f_code.co_name
                except Exception:
                    caller = "?"
                print("[stgev] CLOSE via=%s live=%d filled=%d/%d name=%s"
                      % (caller, len(self), len(v.filled), v.expected, v.name),
                      file=_s.stderr, flush=True)
            return v

        def clear(self):
            if self:
                print("[stgev] CLEAR at_end n=%d names=%s"
                      % (len(self), [e.name for e in list(self.values())[:3]]),
                      file=_s.stderr, flush=True)
            return super().clear()

    orig_init = P.__init__

    def init(self, resolve_moe):
        orig_init(self, resolve_moe)
        self._entries = _LogDict()

    P.__init__ = init
    print("[stgev] entry-event tracing installed", file=_s.stderr, flush=True)


if _STGEV:
    _prev_stg = _HOOKS.get("atom.model_loader.expert_staging")

    def _stg_with_events(mod, _p=_prev_stg):
        if _p is not None:
            _p(mod)
        try:
            _install_stage_events(mod)
        except Exception as exc:
            print("[stgev] hook failed: %r" % (exc,), file=sys.stderr, flush=True)

    _HOOKS["atom.model_loader.expert_staging"] = _stg_with_events


# --------------------------------------------------------------------------
# GLM_SPARSE_DECODE=1 -- give the Triton MLA backend a SPARSE decode path.
#
# ATOM's MLAAttention._forward_decode has three branches; only the third (the
# asm `mla_decode_fwd`, which has no gfx1250 build) honours `is_sparse_mla`.
# Both triton branches -- the shuffled kernel and `decode_attention_fwd` --
# read the DENSE block table and attend over the full context, ignoring the
# indexer's top-k entirely.  Below index_topk that is correct by coincidence
# (top-k == everything); above it the model gets dense attention it was never
# trained for.  The indexer still runs and still fills
# `sparse_kv_indices_buffer`; nothing consumes it.
#
# This hook routes the sparse case to aiter's unified_attention_sparse_mla
# (ALL_DECODE=1), fed from that buffer, and leaves every other case on the
# original implementation.
# --------------------------------------------------------------------------
_SPD = os.environ.get("GLM_SPARSE_DECODE") == "1"


def _install_sparse_decode(mod):
    import sys as _s

    import torch
    from aiter.ops.triton.attention.unified_attention_sparse_mla import (
        unified_attention_sparse_mla,
    )

    MLA = mod.MLAAttention
    orig = MLA._forward_decode
    st = {"n": 0, "sparse": 0, "dense": 0}

    def _forward_decode(self, q, kv_c_and_k_pe_cache, attn_metadata,
                        return_lse=False):
        md = attn_metadata
        buf = getattr(self, "sparse_kv_indices_buffer", None)
        use = (
            getattr(self, "is_sparse_mla", False)
            and buf is not None
            and not return_lse
            and getattr(md, "sparse_kv_indptr", None) is not None
            and int(getattr(md, "max_seqlen_q", 1)) == 1
            and int(getattr(md, "max_seqlen_k", 0)) > int(self.topk_tokens)
        )
        st["n"] += 1
        if not use:
            st["dense"] += 1
            return orig(self, q, kv_c_and_k_pe_cache, attn_metadata, return_lse)
        st["sparse"] += 1
        if st["sparse"] in (1, 78, 780):
            print("[spd] sparse decode #%d q=%s seqlen_k=%d topk=%d"
                  % (st["sparse"], tuple(q.shape), int(md.max_seqlen_k),
                     int(self.topk_tokens)), file=_s.stderr, flush=True)

        num_heads_q = q.shape[1]
        q = self._pad_query_heads(q)
        if self.use_seg_mla:
            q = q[..., : self.kv_lora_rank + self.qk_rope_head_dim]
        if q.dtype.is_floating_point and q.element_size() == 1:
            q = q.to(torch.bfloat16)
        B = q.shape[0]
        o = torch.empty(B, q.shape[1], self.kv_lora_rank,
                        dtype=self.dtype, device=q.device)

        # Same re-view as the prefill adapter: ATOM keeps the MLA cache as a
        # flat [num_token_slots, 1, d] pool, but the kernel takes TILE_SIZE
        # from kv.shape[1].  slot -> (slot // tile, slot % tile) is exactly the
        # addressing it performs, so this is a pure reshape.
        d = self.kv_lora_rank + self.qk_rope_head_dim
        flat = kv_c_and_k_pe_cache.reshape(-1, d)
        n = flat.shape[0]
        tile = 64 if n % 64 == 0 else (16 if n % 16 == 0 else 1)
        kv4 = flat.view(-1, tile, 1, d)

        # CSR -> dense [B, topk] with -1 padding.  Width is pinned to
        # topk_tokens rather than counts.max() so there is no per-layer device
        # sync and the kernel sees one constant shape.
        K = int(self.topk_tokens)
        kvp = md.sparse_kv_indptr[: B + 1].to(torch.int64)
        counts = (kvp[1:] - kvp[:-1]).clamp_min(0)
        ar = torch.arange(K, device=q.device)
        gather = (kvp[:-1].unsqueeze(1) + ar.unsqueeze(0)).clamp(
            0, buf.numel() - 1)
        dense = buf.to(torch.int32)[gather]
        dense = torch.where(ar.unsqueeze(0) < counts.unsqueeze(1), dense,
                            torch.full_like(dense, -1)).contiguous()

        kv_scale = None
        if kv_c_and_k_pe_cache.dtype not in (torch.bfloat16, torch.float16):
            kv_scale = self._k_scale
        unified_attention_sparse_mla(
            q, kv4, o,
            torch.arange(B + 1, dtype=torch.int32, device=q.device),  # cu_seqlens_q
            1,                                   # max_seqlen_q -> ALL_DECODE
            counts.to(torch.int32),              # seqused_k (unused by kernel)
            K,                                   # max_seqlen_k
            self.scale,
            dense,
            torch.zeros((B, 1), dtype=torch.int32, device=q.device),  # dead arg
            self.kv_lora_rank,
            kv_scale,
        )
        o = self._restore_query_heads(o, num_heads_q)
        return self._v_up_proj_and_o_proj(o)

    MLA._forward_decode = _forward_decode

    import atexit

    def _report():
        print("[spd] decode calls: total=%d sparse=%d dense=%d"
              % (st["n"], st["sparse"], st["dense"]),
              file=_s.stderr, flush=True)

    atexit.register(_report)
    print("[spd] sparse decode path installed", file=_s.stderr, flush=True)


if _SPD:
    _prev_am_spd = _HOOKS.get("atom.model_ops.attention_mla")

    def _am_with_spd(mod, _p=_prev_am_spd):
        if _p is not None:
            _p(mod)
        try:
            _install_sparse_decode(mod)
        except Exception as exc:
            print("[spd] hook failed: %r" % (exc,), file=sys.stderr, flush=True)

    _HOOKS["atom.model_ops.attention_mla"] = _am_with_spd


# --------------------------------------------------------------------------
# GLM_IDX_CHECK=1 -- validate the sparse DSA prefill index chain in situ.
#
# Multi-sequence sparse prefill steps produce garbage on gfx1250 while
# single-sequence steps are fine, so every per-sequence offset in the chain is
# suspect. This wraps the last stage (per-row top-k -> global KV slots) and
# asserts the invariants that must hold for each query row:
#   * every selected column lies inside that row's own [rowStart, rowEnd)
#     window (no cross-sequence leakage),
#   * the logits it selected from are finite,
#   * the selections are distinct.
# Reports per sparse-prefill call for the first few calls, then only on
# violation.
# --------------------------------------------------------------------------
_IDXCHK = os.environ.get("GLM_IDX_CHECK") == "1"


def _install_idx_check(mod):
    import sys as _s

    import torch

    st = {"n": 0, "bad": 0}
    orig_conv = mod.triton_convert_req_index_to_global_index_dsa_prefill

    def conv(dsa_qo_indptr, dsa_kv_indptr, token_to_seq_idxs, topk_indices,
             block_tables, cu_seqlens_k, *a, **kw):
        st["n"] += 1
        try:
            n_tok = int(token_to_seq_idxs.shape[0])
            req = token_to_seq_idxs[:n_tok].long()
            cu = cu_seqlens_k.long()
            lo = cu[req].unsqueeze(1)
            hi = cu[req + 1].unsqueeze(1)
            ti = topk_indices[:n_tok].long()
            valid = ti >= 0
            oob = int((valid & ((ti < lo) | (ti >= hi))).sum().item())
            nvalid = valid.sum(1)
            kvlen = (dsa_kv_indptr[1 : n_tok + 1].long()
                     - dsa_kv_indptr[:n_tok].long())
            mismatch = int((nvalid != kvlen).sum().item())
            # duplicates within a row (only over valid entries)
            masked = torch.where(valid, ti, torch.full_like(ti, -1))
            srt, _ = masked.sort(dim=1)
            dup = int(((srt[:, 1:] == srt[:, :-1]) & (srt[:, 1:] >= 0))
                      .sum().item())
            nseq = int(cu.numel() - 1)
            if oob or mismatch or dup:
                st["bad"] += 1
            if st["n"] == 1:
                for sq in range(min(nseq, 3)):
                    rows = (req == sq).nonzero().flatten()
                    if rows.numel() == 0:
                        continue
                    pick = rows[[0, rows.numel() // 2, rows.numel() - 1]]
                    print("[idxchk]   seq%d rows=%s nvalid=%s kvlen=%s"
                          % (sq, pick.tolist(), nvalid[pick].tolist(),
                             kvlen[pick].tolist()), file=_s.stderr, flush=True)
            if st["n"] <= 6 or oob or mismatch or dup:
                print("[idxchk] call#%d nseq=%d n_tok=%d cu=%s | oob=%d "
                      "count_mismatch=%d dup=%d"
                      % (st["n"], nseq, n_tok, cu[: min(6, cu.numel())].tolist(),
                         oob, mismatch, dup), file=_s.stderr, flush=True)
        except Exception as exc:
            print("[idxchk] check failed: %r" % (exc,), file=_s.stderr, flush=True)
        return orig_conv(dsa_qo_indptr, dsa_kv_indptr, token_to_seq_idxs,
                         topk_indices, block_tables, cu_seqlens_k, *a, **kw)

    mod.triton_convert_req_index_to_global_index_dsa_prefill = conv

    # deepseek_v2 imported the symbol by value; rebind there too.
    for mn, m in list(_s.modules.items()):
        if m is None or not mn.startswith("atom."):
            continue
        if getattr(m, "triton_convert_req_index_to_global_index_dsa_prefill",
                   None) is not None and m is not mod:
            m.triton_convert_req_index_to_global_index_dsa_prefill = conv

    print("[idxchk] sparse prefill index validator installed",
          file=_s.stderr, flush=True)


if _IDXCHK:
    _prev_am_ic = _HOOKS.get("atom.model_ops.attention_mla")

    def _am_with_ic(mod, _p=_prev_am_ic):
        if _p is not None:
            _p(mod)
        try:
            _install_idx_check(mod)
        except Exception as exc:
            print("[idxchk] hook failed: %r" % (exc,), file=sys.stderr, flush=True)

    _HOOKS["atom.model_ops.attention_mla"] = _am_with_ic


# --------------------------------------------------------------------------
# GLM_TK_CHECK=1 -- is the truncated top-k window ATOM's row bounds, or -inf
# left inside the window by fp8_mqa_logits?  Wraps top_k_per_row_prefill where
# deepseek_v2 calls it and reports, per row: the [rowStart, rowEnd) width ATOM
# asked for vs how many of those logits are actually finite.
# --------------------------------------------------------------------------
_TKCHK = os.environ.get("GLM_TK_CHECK") == "1"


def _install_tk_check(mod):
    import sys as _s

    import torch

    st = {"n": 0}
    orig = mod.top_k_per_row_prefill

    def wrapped(logits, rowStarts, rowEnds, indices, values, numRows,
                stride0, stride1, *a, **kw):
        st["n"] += 1
        if st["n"] <= 2:
            try:
                n = int(numRows)
                s = rowStarts[:n].long()
                e = rowEnds[:n].long()
                width = (e - s).clamp_min(0)
                cols = torch.arange(logits.shape[1], device=logits.device)
                in_row = (cols.unsqueeze(0) >= s.unsqueeze(1)) & (
                    cols.unsqueeze(0) < e.unsqueeze(1))
                finite = torch.isfinite(logits[:n].float()) & in_row
                nfin = finite.sum(1)
                # sample three rows per distinct rowStart value
                uniq = torch.unique(s)
                print("[tkchk] call#%d numRows=%d logits=%s distinct_rowStarts=%s"
                      % (st["n"], n, tuple(logits.shape), uniq[:6].tolist()),
                      file=_s.stderr, flush=True)
                for u in uniq[:3].tolist():
                    rows = (s == u).nonzero().flatten()
                    pick = rows[[0, rows.numel() // 2, rows.numel() - 1]]
                    print("[tkchk]   start=%d rows=%s width=%s finite_in_window=%s"
                          % (u, pick.tolist(), width[pick].tolist(),
                             nfin[pick].tolist()), file=_s.stderr, flush=True)
            except Exception as exc:
                print("[tkchk] probe failed: %r" % (exc,), file=_s.stderr,
                      flush=True)
        return orig(logits, rowStarts, rowEnds, indices, values, numRows,
                    stride0, stride1, *a, **kw)

    mod.top_k_per_row_prefill = wrapped
    print("[tkchk] top-k window probe installed", file=_s.stderr, flush=True)


if _TKCHK:
    _prev_ds = _HOOKS.get("atom.models.deepseek_v2")

    def _ds_with_tk(mod, _p=_prev_ds):
        if _p is not None:
            _p(mod)
        try:
            _install_tk_check(mod)
        except Exception as exc:
            print("[tkchk] hook failed: %r" % (exc,), file=sys.stderr, flush=True)

    _HOOKS["atom.models.deepseek_v2"] = _ds_with_tk


# --------------------------------------------------------------------------
# GLM_NO_GLUON_MQA=1 -- work around the gfx1250 gluon fp8_mqa_logits kernel.
#
# For a query row whose KV window starts at a nonzero offset (any sequence
# after the first in a multi-sequence sparse prefill step) that kernel only
# scores floor(width / 256) * 256 columns and leaves the ragged tail at the
# -inf fill -- i.e. the newest, causally most important tokens of the row's own
# context are never scored.  Row 0 of such a sequence (width 1) is scored not at
# all.  Clearing the gluon entry point falls the wrapper back to the generic
# triton kernel, whose post-loop masked iteration covers the tail correctly.
# --------------------------------------------------------------------------
_NOGLU = os.environ.get("GLM_NO_GLUON_MQA") == "1"


def _install_no_gluon_mqa(mod):
    import sys as _s

    if getattr(mod, "_gluon_fp8_mqa_logits_kernel", None) is not None:
        mod._gluon_fp8_mqa_logits_kernel = None
        print("[noglu] gluon fp8_mqa_logits disabled in %s" % (mod.__name__,),
              file=_s.stderr, flush=True)


if _NOGLU:
    for _mn in ("aiter.ops.triton.attention.fp8_mqa_logits",
                "aiter.ops.triton.fp8_mqa_logits"):
        _prev_ng = _HOOKS.get(_mn)

        def _with_ng(mod, _p=_prev_ng):
            if _p is not None:
                _p(mod)
            try:
                _install_no_gluon_mqa(mod)
            except Exception as exc:
                print("[noglu] hook failed: %r" % (exc,), file=sys.stderr,
                      flush=True)

        _HOOKS[_mn] = _with_ng


# --------------------------------------------------------------------------
# GLM_MQA_TAIL=1 -- repair the window tail the gfx1250 gluon fp8_mqa_logits
# kernel skips.  See shim/mqa_tail.py.  Supersedes GLM_NO_GLUON_MQA, which
# swapped in the generic triton kernel: that one covers the whole window but is
# numerically wrong on gfx1250 (cos 0.117 vs a float32 replay, against the
# gluon kernel's 1.000000000 on the columns it does write).
# --------------------------------------------------------------------------
_MQATAIL = os.environ.get("GLM_MQA_TAIL") == "1"


def _install_mqa_tail(mod):
    import sys as _s

    from mqa_tail import repair_tail

    orig = mod.fp8_mqa_logits
    st = {"n": 0}

    def wrapped(Q, KV, kv_scales, weights, cu_starts, cu_ends,
                clean_logits=True, **kw):
        out = orig(Q, KV, kv_scales, weights, cu_starts, cu_ends,
                   clean_logits, **kw)
        try:
            if os.environ.get("GLM_MQA_COUNT") == "1":
                import torch as _t
                _n = out.shape[0]
                _c = _t.arange(out.shape[1], device=out.device)
                _iw = ((_c.unsqueeze(0) >= cu_starts[:_n].long().unsqueeze(1))
                       & (_c.unsqueeze(0) < cu_ends[:_n].long()
                          .clamp(max=out.shape[1]).unsqueeze(1)))
                _miss = int((_iw & _t.isneginf(out)).sum())
                st["missing"] = st.get("missing", 0) + _miss
                st["maxmiss"] = max(st.get("maxmiss", 0), _miss)
            repair_tail(out, Q, KV, kv_scales, weights, cu_starts, cu_ends)
        except Exception as exc:
            if st["n"] < 3:
                print("[mqatail] repair failed: %r" % (exc,), file=_s.stderr,
                      flush=True)
        st["n"] += 1
        return out

    mod.fp8_mqa_logits = wrapped
    print("[mqatail] fp8_mqa_logits tail repair installed", file=_s.stderr,
          flush=True)
    import atexit as _ax
    _ax.register(lambda: print(
        "[mqatail] SUMMARY calls=%d columns_needing_repair=%s (max %s per call)"
        % (st["n"], st.get("missing", "n/a"), st.get("maxmiss", "n/a")),
        file=_s.stderr, flush=True))


if _MQATAIL:
    _prev_ds_mt = _HOOKS.get("atom.models.deepseek_v2")

    def _ds_with_mt(mod, _p=_prev_ds_mt):
        if _p is not None:
            _p(mod)
        try:
            _install_mqa_tail(mod)
        except Exception as exc:
            print("[mqatail] hook failed: %r" % (exc,), file=sys.stderr,
                  flush=True)

    _HOOKS["atom.models.deepseek_v2"] = _ds_with_mt


# --------------------------------------------------------------------------
# GLM_TSM_DIFF=1 -- why is the triton unified_attention_sparse_mla prefill ~5.7
# points worse on gsm8k than the torch reference?  Runs BOTH on every real
# sparse-prefill call (same q, same KV, same top-k CSR), keeps the triton result
# as the live output so the trajectory matches the measured arm, and reports how
# they differ.  Budgeted: after GLM_TSM_DIFF_CALLS calls it stops comparing.
# --------------------------------------------------------------------------
_TSMDIFF = os.environ.get("GLM_TSM_DIFF") == "1"
_TSMDIFF_N = int(os.environ.get("GLM_TSM_DIFF_CALLS", "240"))


_TSMDIFF_DONE = []


def _install_tsm_diff(aiter_mod):
    import sys as _s

    import torch

    if _TSMDIFF_DONE:
        return
    _TSMDIFF_DONE.append(1)
    ref = _make_custom_impls()["mla_prefill_asm_fwd"]

    st = {"n": 0, "cmp": 0, "cos_sum": 0.0, "rel_sum": 0.0,
          "worst_cos": 1.0, "bad_tok": 0, "tot_tok": 0, "mag_sum": 0.0}

    # bind the currently-installed (triton) implementation
    live = None
    for mn, m in list(_s.modules.items()):
        if m is None or (mn != "aiter" and not mn.startswith("aiter.")):
            continue
        f = getattr(m, "mla_prefill_asm_fwd", None)
        if f is not None:
            live = f
            break
    if live is None:
        print("[tsmdiff] no mla_prefill_asm_fwd bound", file=_s.stderr, flush=True)
        return

    def both(q, kv_buffer, qo_indptr, kv_indptr, kv_indices, kv_last_page_lens,
             max_seqlen_q, sm_scale, logits, attn_lse, *a, **kw):
        st["n"] += 1
        if st["cmp"] >= _TSMDIFF_N:
            return live(q, kv_buffer, qo_indptr, kv_indptr, kv_indices,
                        kv_last_page_lens, max_seqlen_q, sm_scale, logits,
                        attn_lse, *a, **kw)
        ref_buf = torch.empty_like(logits)
        # the torch reference writes a real LSE (the triton adapter zeroes it);
        # mla_prefill_fwd discards it either way, but it must be a tensor
        ref_lse = None if attn_lse is None else torch.empty_like(attn_lse)
        ref(q, kv_buffer, qo_indptr, kv_indptr, kv_indices, kv_last_page_lens,
            max_seqlen_q, sm_scale, ref_buf, ref_lse, *a, **kw)
        r = ref_buf.reshape(logits.shape[0], -1).float()
        live(q, kv_buffer, qo_indptr, kv_indptr, kv_indices, kv_last_page_lens,
             max_seqlen_q, sm_scale, logits, attn_lse, *a, **kw)
        t = logits.reshape(logits.shape[0], -1).float()
        del ref_buf

        st["cmp"] += 1
        cos_all = torch.nn.functional.cosine_similarity(
            t.flatten().double(), r.flatten().double(), dim=0).item()
        rel = ((t - r).norm() / r.norm().clamp_min(1e-30)).item()
        mag = (t.norm() / r.norm().clamp_min(1e-30)).item()
        ct = torch.nn.functional.cosine_similarity(t, r, dim=1)
        st["cos_sum"] += cos_all
        st["rel_sum"] += rel
        st["mag_sum"] += mag
        st["worst_cos"] = min(st["worst_cos"], float(ct.min()))
        st["bad_tok"] += int((ct < 0.99).sum())
        st["tot_tok"] += int(ct.numel())
        if st["cmp"] <= 5 or st["cmp"] % 60 == 0:
            k = int(kv_indptr[1] - kv_indptr[0]) if kv_indptr.numel() > 1 else -1
            print("[tsmdiff] call#%d rows=%d k0=%d maxq=%s | cos=%.9f rel=%.3e "
                  "mag=%.6f worst_tok_cos=%.6f frac_tok<0.99=%.4f"
                  % (st["cmp"], t.shape[0], k, max_seqlen_q, cos_all, rel, mag,
                     float(ct.min()), float((ct < 0.99).float().mean())),
                  file=_s.stderr, flush=True)

    for mn, m in list(_s.modules.items()):
        if m is None or (mn != "aiter" and not mn.startswith("aiter.")):
            continue
        if getattr(m, "mla_prefill_asm_fwd", None) is not None:
            setattr(m, "mla_prefill_asm_fwd", both)

    import atexit

    def _rep():
        c = max(st["cmp"], 1)
        print("[tsmdiff] SUMMARY calls=%d compared=%d mean_cos=%.9f "
              "mean_rel=%.4e mean_mag=%.6f worst_tok_cos=%.6f "
              "tokens<0.99=%d/%d (%.3f%%)"
              % (st["n"], st["cmp"], st["cos_sum"] / c, st["rel_sum"] / c,
                 st["mag_sum"] / c, st["worst_cos"], st["bad_tok"],
                 st["tot_tok"], 100.0 * st["bad_tok"] / max(st["tot_tok"], 1)),
              file=_s.stderr, flush=True)

    atexit.register(_rep)
    print("[tsmdiff] triton-vs-torchref differ installed (budget %d calls)"
          % (_TSMDIFF_N,), file=_s.stderr, flush=True)


if _TSMDIFF:
    _prev_a_diff = _HOOKS.get("aiter")

    def _aiter_with_diff(mod, _p=_prev_a_diff):
        if _p is not None:
            _p(mod)
        try:
            _install_tsm_diff(mod)
        except Exception as exc:
            print("[tsmdiff] hook failed: %r" % (exc,), file=sys.stderr, flush=True)

    _HOOKS["aiter"] = _aiter_with_diff


# --------------------------------------------------------------------------
# GLM_ATTN_NOISE=<rel> -- calibrate how much gsm8k accuracy a given relative
# perturbation of the sparse-prefill attention output costs.
#
# The torch "reference" mla_prefill_asm_fwd runs the whole attention in fp32;
# the aiter triton kernel is bf16 and lands 1.6e-3 rel L2 away from it, with
# zero outlier rows (cos 0.999998, worst per-token cos 0.999995) -- i.e. about
# one bf16 ULP, which is what a correct bf16 kernel should give. So the question
# is not "is the triton kernel wrong" but "is 5.7 gsm8k points a plausible cost
# of 1.6e-3 of noise". This adds exactly that much Gaussian noise to the fp32
# reference's output and re-measures.
# --------------------------------------------------------------------------
_ATTNNOISE = float(os.environ.get("GLM_ATTN_NOISE", "0") or 0)


def _install_attn_noise(aiter_mod):
    import sys as _s

    import torch

    if getattr(_install_attn_noise, "_done", False):
        return
    _install_attn_noise._done = True

    live = None
    for mn, m in list(_s.modules.items()):
        if m is None or (mn != "aiter" and not mn.startswith("aiter.")):
            continue
        f = getattr(m, "mla_prefill_asm_fwd", None)
        if f is not None:
            live = f
            break
    if live is None:
        return

    st = {"n": 0}

    def noisy(q, kv_buffer, qo_indptr, kv_indptr, kv_indices, kv_last_page_lens,
              max_seqlen_q, sm_scale, logits, attn_lse, *a, **kw):
        live(q, kv_buffer, qo_indptr, kv_indptr, kv_indices, kv_last_page_lens,
             max_seqlen_q, sm_scale, logits, attn_lse, *a, **kw)
        # relative per-element Gaussian noise scaled to the row RMS, so the
        # induced rel L2 over the call is _ATTNNOISE by construction
        f = logits.float()
        rms = f.pow(2).mean(dim=-1, keepdim=True).sqrt()
        logits.copy_((f + torch.randn_like(f) * rms * _ATTNNOISE).to(logits.dtype))
        st["n"] += 1
        if st["n"] in (1, 100):
            print("[attnnoise] call#%d injecting rel=%.3e" % (st["n"], _ATTNNOISE),
                  file=_s.stderr, flush=True)

    for mn, m in list(_s.modules.items()):
        if m is None or (mn != "aiter" and not mn.startswith("aiter.")):
            continue
        if getattr(m, "mla_prefill_asm_fwd", None) is not None:
            setattr(m, "mla_prefill_asm_fwd", noisy)
    print("[attnnoise] rel=%.3e injected into sparse prefill output"
          % (_ATTNNOISE,), file=_s.stderr, flush=True)


if _ATTNNOISE > 0:
    _prev_a_noise = _HOOKS.get("aiter")

    def _aiter_with_noise(mod, _p=_prev_a_noise):
        if _p is not None:
            _p(mod)
        try:
            _install_attn_noise(mod)
        except Exception as exc:
            print("[attnnoise] hook failed: %r" % (exc,), file=sys.stderr, flush=True)

    _HOOKS["aiter"] = _aiter_with_noise


# --------------------------------------------------------------------------
# GLM_SAMPLE_CHECK=1 -- why does gfx1250 emit stray token 0 ('!') mid-generation
# (207/300 gsm8k outputs, vs 0/300 on gfx950)?
#
# ATOM's greedy path is mixed_sample_outer_exponential(out, logits, exp, T),
# which for T==0 reduces to logits.argmax(-1). argmax returns 0 for three very
# different reasons: an all-NaN row, a flat/all-equal row, or token 0 honestly
# winning. This wraps the op and, for every row that sampled 0, dumps enough of
# the logit row to tell those apart.
# --------------------------------------------------------------------------
_SMPCHK = os.environ.get("GLM_SAMPLE_CHECK") == "1"
_SMPCHK_N = int(os.environ.get("GLM_SAMPLE_CHECK_N", "24"))


def _install_sample_check(mod):
    import sys as _s

    import torch

    if getattr(_install_sample_check, "_done", False):
        return
    _install_sample_check._done = True
    orig = mod.mixed_sample_outer_exponential
    st = {"calls": 0, "rows": 0, "zero": 0, "rep": 0,
          "nan": 0, "flat": 0, "honest": 0}

    def wrapped(out, input, exponentials, temperature, eps=1e-10):
        orig(out, input, exponentials, temperature, eps=eps)
        st["calls"] += 1
        st["rows"] += int(out.numel())
        z = (out == 0)
        nz = int(z.sum())
        if not nz:
            return
        st["zero"] += nz
        for r in z.nonzero().flatten().tolist():
            row = input[r].float()
            nan = int(torch.isnan(row).sum())
            inf = int(torch.isinf(row).sum())
            std = float(row.std())
            if nan:
                st["nan"] += 1
                kind = "NaN"
            elif std < 1e-6:
                st["flat"] += 1
                kind = "FLAT"
            else:
                st["honest"] += 1
                kind = "argmax0"
            try:
                blowup_report("%s row=%d" % (kind, r))
            except Exception:
                pass
            try:
                kvwatch_report("%s row=%d" % (kind, r))
            except Exception:
                pass
            if st["rep"] < _SMPCHK_N:
                st["rep"] += 1
                v, i = row.topk(4)
                print("[smpchk] %s call#%d row=%d/%d nan=%d inf=%d std=%.4f "
                      "logit0=%.4f top4=%s @%s T=%.3g"
                      % (kind, st["calls"], r, int(out.numel()), nan, inf, std,
                         float(row[0]), [round(float(x), 3) for x in v],
                         i.tolist(), float(temperature.reshape(-1)[r])),
                      file=_s.stderr, flush=True)

    mod.mixed_sample_outer_exponential = wrapped

    import atexit

    def _rep():
        print("[smpchk] SUMMARY calls=%d rows=%d sampled_token0=%d "
              "(NaN=%d FLAT=%d honest_argmax0=%d)  rate=%.4f%%"
              % (st["calls"], st["rows"], st["zero"], st["nan"], st["flat"],
                 st["honest"], 100.0 * st["zero"] / max(st["rows"], 1)),
              file=_s.stderr, flush=True)

    atexit.register(_rep)
    print("[smpchk] sampler token-0 probe installed", file=_s.stderr, flush=True)


if _SMPCHK:
    _prev_smp = _HOOKS.get("atom.model_ops.sampler")

    def _smp_hook(mod, _p=_prev_smp):
        if _p is not None:
            _p(mod)
        try:
            _install_sample_check(mod)
        except Exception as exc:
            print("[smpchk] hook failed: %r" % (exc,), file=sys.stderr, flush=True)

    _HOOKS["atom.model_ops.sampler"] = _smp_hook


# --------------------------------------------------------------------------
# GLM_KVGUARD=1 -- find where the ~1e38 poison in the sparse-prefill attention
# output comes from.
#
# Symptom: for some prompts one layer's attention output is a single vector of
# magnitude ~8.4e37, bit-identical across every query row, which saturates the
# residual stream; every later layer then emits exactly 0, the final logits are
# all 0, and argmax returns token 0 -- the stray '!'.
#
# Since out = sum_k p_k V_k and every row returns the same vector, one gathered
# KV slot must itself be ~1e38 and win every row's softmax. This wraps the
# sparse-prefill attention, and whenever the output blows up, reports which
# gathered slots are huge and whether they lie inside the range the KV writer
# actually touched.
# --------------------------------------------------------------------------
_KVGUARD = os.environ.get("GLM_KVGUARD") == "1"


def _install_kvguard(aiter_mod):
    import sys as _s

    import torch

    if getattr(_install_kvguard, "_done", False):
        return
    _install_kvguard._done = True

    live = None
    for mn, m in list(_s.modules.items()):
        if m is None or (mn != "aiter" and not mn.startswith("aiter.")):
            continue
        if getattr(m, "mla_prefill_asm_fwd", None) is not None:
            live = getattr(m, "mla_prefill_asm_fwd")
            break
    if live is None:
        return
    st = {"n": 0, "hit": 0}

    def guarded(q, kv_buffer, qo_indptr, kv_indptr, kv_indices, kv_last_page_lens,
                max_seqlen_q, sm_scale, logits, attn_lse, *a, **kw):
        live(q, kv_buffer, qo_indptr, kv_indptr, kv_indices, kv_last_page_lens,
             max_seqlen_q, sm_scale, logits, attn_lse, *a, **kw)
        st["n"] += 1
        mx = float(logits.abs().max())
        if mx < 1e30 or st["hit"] >= 4:
            return
        st["hit"] += 1
        d = kv_buffer.shape[-1]
        flat = kv_buffer.reshape(-1, d)
        idx = kv_indices.to(torch.int64)
        valid = idx[idx >= 0]
        # Scan the CACHE (a few 10k slots), never the GATHER: kv_indices holds
        # num_tokens * topk entries, so materialising flat[valid] is millions of
        # rows and OOMs at gpu-memory-utilization 0.985 -- which is what killed
        # the first version of this guard mid-diagnosis.
        cache_rowmax = flat.abs().amax(dim=1).float()
        cache_bad = (cache_rowmax > 1e30).nonzero().flatten()
        # which of the huge slots were actually selected by the top-k
        if cache_bad.numel():
            sel = torch.isin(cache_bad, valid)
            bad = cache_bad[sel]
        else:
            bad = cache_bad
        print("[kvguard] call#%d out_max=%.4g  q_max=%.4g  |kv_indices|=%d "
              "distinct_slots=%d" % (st["n"], mx, float(q.abs().max()),
                                     int(valid.numel()),
                                     int(torch.unique(valid).numel())),
              file=_s.stderr, flush=True)
        print("[kvguard]   huge cache slots THAT WERE SELECTED: %d ; ids %s"
              % (int(bad.numel()), bad[:6].tolist()),
              file=_s.stderr, flush=True)
        print("[kvguard]   WHOLE cache rows >1e30: %d of %d ; first ids %s ; "
              "cache_max=%.4g" % (int(cache_bad.numel()), int(flat.shape[0]),
                                  cache_bad[:6].tolist(), float(cache_rowmax.max())),
              file=_s.stderr, flush=True)
        if valid.numel():
            print("[kvguard]   gathered slot range [%d, %d] ; written-range guess "
                  "max_slot_in_indices=%d"
                  % (int(valid.min()), int(valid.max()), int(valid.max())),
                  file=_s.stderr, flush=True)

    for mn, m in list(_s.modules.items()):
        if m is None or (mn != "aiter" and not mn.startswith("aiter.")):
            continue
        if getattr(m, "mla_prefill_asm_fwd", None) is not None:
            setattr(m, "mla_prefill_asm_fwd", guarded)
    print("[kvguard] sparse-prefill blow-up guard installed", file=_s.stderr,
          flush=True)


if _KVGUARD:
    _prev_a_kvg = _HOOKS.get("aiter")

    def _aiter_with_kvg(mod, _p=_prev_a_kvg):
        if _p is not None:
            _p(mod)
        try:
            _install_kvguard(mod)
        except Exception as exc:
            print("[kvguard] hook failed: %r" % (exc,), file=sys.stderr, flush=True)

    _HOOKS["aiter"] = _aiter_with_kvg


# --------------------------------------------------------------------------
# GLM_BLOWUP=1 -- locate the layer where the residual stream explodes.
#
# The stray '!' is token 0 chosen by argmax over an all-zero logit row (24 of 26
# cases in a 20-question run; the rest NaN). Zeros downstream of a saturated
# residual is exactly what the tap dump showed for a prefill blow-up: one
# layer's attention emits ~8.4e37, every later layer then emits exactly 0. But
# the prefill guard does not fire during normal runs, so the poison usually
# appears in DECODE. This keeps a device-side per-layer running max (no host
# sync in the hot path) and dumps the profile the moment a row samples token 0.
# --------------------------------------------------------------------------
_BLOWUP = os.environ.get("GLM_BLOWUP") == "1"
_BLOWUP_ST = {}


def _install_blowup(torch_mod):
    torch = torch_mod
    st = {"L": 0, "buf": None, "n": 0}
    _BLOWUP_ST["st"] = st

    def hook(module, args, output):
        cls = type(module).__name__
        if "Embedding" in cls:
            o = output[0] if isinstance(output, (tuple, list)) else output
            if hasattr(o, "detach"):
                st["emb_t"] = o.detach().abs().max().float()
            return
        if cls != "DeepseekV2DecoderLayer":
            return
        x = None
        if isinstance(output, (tuple, list)) and len(output) > 1 and \
                hasattr(output[1], "detach"):
            x = output[1]                      # residual stream
        elif isinstance(output, (tuple, list)):
            x = output[0]
        else:
            x = output
        if not hasattr(x, "detach"):
            return
        if st["buf"] is None:
            # Allocate OUTSIDE inference mode: a tensor created inside it is an
            # inference tensor, and updating one from normal mode later raises.
            with torch.inference_mode(False):
                st["buf"] = torch.zeros(78, device=x.device, dtype=torch.float32)
        L = st["L"] % 78
        if L == 0:
            # New forward. Without this the buffer is a running max over every
            # step since the last report, which silently imported the warmup
            # dummy run's NaNs into the profile.
            st["buf"].zero_()
        st["buf"][L] = torch.maximum(st["buf"][L],
                                     x.detach().abs().max().float())
        st["L"] += 1

    torch_mod.nn.modules.module.register_module_forward_hook(hook)
    print("[blowup] per-layer residual magnitude tracker installed",
          file=sys.stderr, flush=True)


def blowup_report(tag):
    """Called from the sampler probe when a row samples token 0."""
    st = _BLOWUP_ST.get("st")
    if not st or st.get("buf") is None or st["n"] >= 6:
        return
    st["n"] += 1
    v = st["buf"].detach().cpu().numpy()
    first = None
    for i, x in enumerate(v):
        if x > 1e30:
            first = i
            break
    prof = " ".join("L%d=%.3g" % (i, v[i]) for i in range(0, 78, 6))
    e = st.get("emb_t")
    if e is not None:
        prof = ("emb=%.4g " % float(e.cpu())) + prof
    print("[blowup] %s  first_layer_over_1e30=%s  max=%.4g\n[blowup]   %s"
          % (tag, first, float(v.max()), prof), file=sys.stderr, flush=True)
    if first is not None and first > 0:
        print("[blowup]   neighbours: L%d=%.4g L%d=%.4g L%d=%.4g"
              % (first - 1, v[first - 1], first, v[first],
                 min(first + 1, 77), v[min(first + 1, 77)]),
              file=sys.stderr, flush=True)
    st["buf"].zero_()


if _BLOWUP:
    _prev_t_bu = _HOOKS.get("torch")

    def _torch_with_blowup(mod, _p=_prev_t_bu):
        if _p is not None:
            _p(mod)
        try:
            _install_blowup(mod)
        except Exception as exc:
            print("[blowup] hook failed: %r" % (exc,), file=sys.stderr, flush=True)

    _HOOKS["torch"] = _torch_with_blowup


# --------------------------------------------------------------------------
# GLM_KVSCAN=1 / GLM_ZERO_KV=1 -- is the ~1e38 poison a read of never-written
# KV cache memory?
#
# The stray '!' traces back to one KV entry of magnitude ~1e38 dominating the
# softmax. If ATOM allocates the paged KV cache with torch.empty (uninitialized
# device memory), any read of a slot the writer has not touched returns whatever
# was there -- and a random bf16 bit pattern is huge far more often than small.
#
# KVSCAN reports, on the first real forward, how much of the freshly allocated
# cache is already out of range. ZERO_KV additionally zeroes it, which should
# make the '!' disappear if that is the source.
# --------------------------------------------------------------------------
_KVSCAN = os.environ.get("GLM_KVSCAN") == "1"
_ZEROKV = os.environ.get("GLM_ZERO_KV") == "1"


def _install_kvscan(mod):
    import sys as _s

    import torch

    if getattr(_install_kvscan, "_done", False):
        return
    _install_kvscan._done = True
    orig = mod.MLAAttention.forward_impl
    state = {"scanned": False}

    def forward_impl(self, *a, **kw):
        if not state["scanned"]:
            try:
                from atom.utils.forward_context import get_forward_context
                fc = get_forward_context()
                if not fc.context.is_dummy_run:
                    state["scanned"] = True
                    tot = huge = nan = 0
                    mx = 0.0
                    nlayer = 0
                    for name, obj in sorted(fc.kv_cache_data.items()):
                        t = getattr(obj, "k_cache", None)
                        if t is None or not hasattr(t, "numel") or t.numel() == 0:
                            continue
                        nlayer += 1
                        f = t.detach().float()
                        tot += int(t.numel())
                        huge += int((f.abs() > 1e30).sum())
                        nan += int(torch.isnan(f).sum())
                        m = float(f.abs().max())
                        mx = max(mx, m)
                    print("[kvscan] fresh KV cache: layers=%d elems=%d "
                          "|v|>1e30: %d (%.4f%%)  NaN: %d  max|v|=%.4g"
                          % (nlayer, tot, huge, 100.0 * huge / max(tot, 1),
                             nan, mx), file=_s.stderr, flush=True)
                    if _ZEROKV:
                        for name, obj in fc.kv_cache_data.items():
                            t = getattr(obj, "k_cache", None)
                            if t is not None and hasattr(t, "zero_"):
                                t.zero_()
                        print("[kvscan] KV cache zeroed", file=_s.stderr,
                              flush=True)
            except Exception as exc:
                state["scanned"] = True
                print("[kvscan] scan failed: %r" % (exc,), file=_s.stderr,
                      flush=True)
        return orig(self, *a, **kw)

    mod.MLAAttention.forward_impl = forward_impl
    print("[kvscan] installed (zero=%s)" % (_ZEROKV,), file=_s.stderr, flush=True)


if _KVSCAN or _ZEROKV:
    _prev_am_ks = _HOOKS.get("atom.model_ops.attention_mla")

    def _am_with_ks(mod, _p=_prev_am_ks):
        if _p is not None:
            _p(mod)
        try:
            _install_kvscan(mod)
        except Exception as exc:
            print("[kvscan] hook failed: %r" % (exc,), file=sys.stderr, flush=True)

    _HOOKS["atom.model_ops.attention_mla"] = _am_with_ks


# --------------------------------------------------------------------------
# GLM_KVWATCH=1 -- is the KV cache poisoned at the moment the logits go flat?
#
# The fresh cache is all zeros (0 of 1.02e9 elements out of range), so the ~1e38
# value is not uninitialised memory -- something writes it during the run, or
# the attention kernel manufactures it from clean inputs. This scans every
# layer's cache at the instant a row samples token 0 and reports where the huge
# entries are, which separates "the KV writer stored garbage" from "the kernel
# blew up on clean data".
# --------------------------------------------------------------------------
_KVWATCH = os.environ.get("GLM_KVWATCH") == "1"
_KVW = {}


def _install_kvwatch(mod):
    import sys as _s

    orig = mod.MLAAttention.forward_impl
    got = {"v": False}

    def forward_impl(self, *a, **kw):
        if not got["v"]:
            try:
                from atom.utils.forward_context import get_forward_context
                fc = get_forward_context()
                if not fc.context.is_dummy_run:
                    _KVW["caches"] = fc.kv_cache_data
                    got["v"] = True
                    print("[kvwatch] captured %d kv caches"
                          % (len(fc.kv_cache_data),), file=_s.stderr, flush=True)
            except Exception:
                pass
        return orig(self, *a, **kw)

    mod.MLAAttention.forward_impl = forward_impl
    print("[kvwatch] installed", file=_s.stderr, flush=True)


def kvwatch_report(tag):
    """Called from the sampler probe when a row samples token 0."""
    import sys as _s

    import torch

    caches = _KVW.get("caches")
    if caches is None or _KVW.get("n", 0) >= 3:
        return
    _KVW["n"] = _KVW.get("n", 0) + 1
    hits = []
    total_huge = 0
    for name, obj in sorted(caches.items(),
                            key=lambda kv: int(kv[0].split("_")[-1])):
        t = getattr(obj, "k_cache", None)
        if t is None or not hasattr(t, "numel") or t.numel() == 0:
            continue
        f = t.detach().float()
        rowmax = f.reshape(-1, f.shape[-1]).abs().amax(dim=1)
        bad = (rowmax > 1e30).nonzero().flatten()
        if bad.numel():
            total_huge += int(bad.numel())
            if len(hits) < 6:
                hits.append((name, int(bad.numel()), bad[:4].tolist(),
                             float(rowmax.max())))
    print("[kvwatch] %s: cache slots with |v|>1e30 = %d ; layers hit: %s"
          % (tag, total_huge,
             [(h[0], h[1], h[2], "%.3g" % h[3]) for h in hits]),
          file=_s.stderr, flush=True)


if _KVWATCH:
    _prev_am_kw = _HOOKS.get("atom.model_ops.attention_mla")

    def _am_with_kw(mod, _p=_prev_am_kw):
        if _p is not None:
            _p(mod)
        try:
            _install_kvwatch(mod)
        except Exception as exc:
            print("[kvwatch] hook failed: %r" % (exc,), file=sys.stderr, flush=True)

    _HOOKS["atom.model_ops.attention_mla"] = _am_with_kw


# --------------------------------------------------------------------------
# GLM_ZERO_DECBUF=1 -- is the decode split-K workspace the source of the ~1e38?
#
# TritonMLAMetadataBuilder allocates
#     triton_attn_logits = torch.empty(max_bs, heads, num_kv_splits, lora+1)
#     triton_lse         = torch.empty(max_bs, heads)
# with torch.EMPTY, and decode_attention_fwd uses attn_logits as the split-K
# partial accumulator: [..., :-1] is the partial numerator and [..., -1] the
# per-split running max. The reduce pass recombines splits as
# exp(m_i - m_max) * acc_i, so a single unwritten/stale m_i that is much larger
# than the true max overflows straight to ~1e38 -- which is exactly the
# signature: one layer's output saturates, every later layer emits 0, the logits
# go flat, and argmax returns token 0 ('!').
#
# Zeroing both buffers before every call is a diagnostic, not a fix: if the rate
# drops the kernel is reading slots it never wrote.
# --------------------------------------------------------------------------
_ZDEC = os.environ.get("GLM_ZERO_DECBUF") == "1"


def _install_zero_decbuf(mod):
    import sys as _s

    if getattr(_install_zero_decbuf, "_done", False):
        return
    _install_zero_decbuf._done = True
    orig = mod.decode_attention_fwd
    st = {"n": 0}

    fill = float(os.environ.get("GLM_DECBUF_FILL", "0"))

    def wrapped(q, k_buffer, v_buffer, o, lse, req_to_token, b_seq_len,
                attn_logits, *a, **kw):
        # Positive control: filling with a large sentinel costs the same work as
        # zeroing, so timing is matched. If the kernel only ever reads split
        # slots it has written, the fill value cannot matter; if it reads
        # unwritten slots, a big sentinel should make the blow-ups far worse.
        attn_logits.fill_(fill)
        lse.fill_(fill)
        st["n"] += 1
        if st["n"] == 1:
            print("[zdec] filling decode split-K workspace with %g: "
                  "attn_logits%s lse%s"
                  % (fill, tuple(attn_logits.shape), tuple(lse.shape)),
                  file=_s.stderr, flush=True)
        return orig(q, k_buffer, v_buffer, o, lse, req_to_token, b_seq_len,
                    attn_logits, *a, **kw)

    mod.decode_attention_fwd = wrapped
    print("[zdec] installed", file=_s.stderr, flush=True)


if _ZDEC:
    _prev_md = _HOOKS.get("aiter.ops.triton.attention.mla_decode")

    def _md_with_z(mod, _p=_prev_md):
        if _p is not None:
            _p(mod)
        try:
            _install_zero_decbuf(mod)
        except Exception as exc:
            print("[zdec] hook failed: %r" % (exc,), file=sys.stderr, flush=True)

    _HOOKS["aiter.ops.triton.attention.mla_decode"] = _md_with_z


# --------------------------------------------------------------------------
# GLM_NANHUNT=1 -- find the FIRST module that emits NaN/1e30.
#
# The guard caught layer 24's attention with q_max=nan on entry while the KV
# cache was clean, so the poison is made upstream, in the q path
# (q_a_proj -> q_a_layernorm -> q_b_proj -> absorption BMM vs W_K -> rope).
# Rather than guess which, walk every module in forward order and report the
# first one whose output is already bad. One prefill of one prompt, so the
# per-module sync is affordable.
# --------------------------------------------------------------------------
_NANHUNT = os.environ.get("GLM_NANHUNT") == "1"


def _install_nanhunt(torch_mod):
    torch = torch_mod
    st = {"rep": 0, "layer": -1}

    def hook(module, args, output):
        if st["rep"] >= 10:
            return
        try:
            from atom.utils.forward_context import get_forward_context
            if get_forward_context().context.is_dummy_run:
                return
        except Exception:
            pass
        cls = type(module).__name__
        if cls == "DeepseekV2DecoderLayer":
            st["layer"] += 1
            return
        xs = output if isinstance(output, (tuple, list)) else (output,)
        for j, x in enumerate(xs):
            if not hasattr(x, "detach") or not hasattr(x, "is_floating_point"):
                continue
            if not x.is_floating_point() or x.numel() == 0:
                continue
            # Only the real prefill. ATOM's cudagraph capture runs bs=1 forwards
            # on uninitialised inputs that is_dummy_run does not flag, and those
            # are full of NaN/huge values that swamp the report.
            if x.dim() < 2 or x.shape[0] <= 2048:
                continue
            f = x.detach()
            nan = bool(torch.isnan(f).any())
            big = False if nan else bool(f.abs().max() > 1e30)
            if nan or big:
                st["rep"] += 1
                print("[nanhunt] #%d after_layer=%d %s (%s) out[%d] shape=%s "
                      "nan=%s big=%s"
                      % (st["rep"], st["layer"],
                         getattr(module, "prefix", "?"), cls, j,
                         tuple(f.shape), nan, big),
                      file=sys.stderr, flush=True)
                break

    torch_mod.nn.modules.module.register_module_forward_hook(hook)
    print("[nanhunt] first-bad-module hunter installed", file=sys.stderr,
          flush=True)


if _NANHUNT:
    _prev_t_nh = _HOOKS.get("torch")

    def _torch_with_nh(mod, _p=_prev_t_nh):
        if _p is not None:
            _p(mod)
        try:
            _install_nanhunt(mod)
        except Exception as exc:
            print("[nanhunt] hook failed: %r" % (exc,), file=sys.stderr, flush=True)

    _HOOKS["torch"] = _torch_with_nh


# --------------------------------------------------------------------------
# GLM_BMMCHK=1 -- is the absorption BMM manufacturing the poison?
#
# nanhunt says every MODULE up to layers.24.self_attn.o_proj is clean, and the
# guard says the absorbed q handed to attention is already NaN. The only step
# between them is batched_gemm_a16wfp4(q_nope, W_K, W_K_scale) -- a bare op, not
# a module, so nothing has looked at it. This checks its inputs and its output
# on every call and reports the first bad one, which separates "the kernel makes
# NaN from clean inputs" from "W_K/q_nope were already bad".
# --------------------------------------------------------------------------
_BMMCHK = os.environ.get("GLM_BMMCHK") == "1"


def _install_bmmchk(mod):
    import sys as _s

    import torch

    if getattr(_install_bmmchk, "_done", False):
        return
    _install_bmmchk._done = True
    orig = mod.batched_gemm_a16wfp4
    st = {"n": 0, "rep": 0}

    def chk(a, w, wscale, *args, **kw):
        out = orig(a, w, wscale, *args, **kw)
        st["n"] += 1
        if st["rep"] >= 4 or a.shape[1] <= 2048:
            return out
        o = out[0] if isinstance(out, (tuple, list)) else out
        bad = bool(torch.isnan(o).any()) or bool(o.abs().max() > 1e30)
        if bad:
            st["rep"] += 1
            def d(t, nm):
                if not hasattr(t, "abs"):
                    return "%s=<%r>" % (nm, type(t).__name__)
                tf = t.detach()
                if tf.dtype in (torch.uint8, torch.int8, torch.int32):
                    return "%s[%s,%s] max_u8=%d" % (nm, tuple(tf.shape),
                                                    tf.dtype, int(tf.max()))
                return "%s[%s,%s] nan=%s max=%.4g" % (
                    nm, tuple(tf.shape), tf.dtype,
                    bool(torch.isnan(tf.float()).any()),
                    float(tf.float().abs().max()))
            print("[bmmchk] call#%d BAD OUTPUT  %s | %s | %s | out nan=%s max=%.4g"
                  % (st["n"], d(a, "q_nope"), d(w, "W_K"), d(wscale, "W_K_scale"),
                     bool(torch.isnan(o).any()), float(o.float().abs().max())),
                  file=_s.stderr, flush=True)
            # Direct test: is the KERNEL deterministically bad in this process,
            # or did this one OUTPUT get corrupted? Re-run the identical call,
            # then again on cloned inputs in fresh memory.
            try:
                o2 = orig(a, w, wscale, *args, **kw)
                o2 = o2[0] if isinstance(o2, (tuple, list)) else o2
                a3, w3, s3 = a.clone(), w.clone(), wscale.clone()
                o3 = orig(a3, w3, s3, *args, **kw)
                o3 = o3[0] if isinstance(o3, (tuple, list)) else o3
                import aiter.ops.triton.gemm.batched.batched_gemm_a16wfp4 as _M
                cfg, _unused = _M._get_config(a.shape[1], w.shape[1], w.shape[2])
                print("[bmmchk]   RERUN same tensors: nan=%d | RERUN cloned: nan=%d"
                      " | a_still_finite=%s"
                      % (int(torch.isnan(o2).sum()), int(torch.isnan(o3).sum()),
                         not bool(torch.isnan(a.float()).any())),
                      file=_s.stderr, flush=True)
                print("[bmmchk]   in-model config: %s"
                      % ({k: cfg[k] for k in sorted(cfg)},),
                      file=_s.stderr, flush=True)
                # WHERE are the NaNs? 2568 of them in a (2568,64,512) output is
                # exactly one per row, so the (b, n) pattern is the tell.
                pos = torch.isnan(o).nonzero()
                if pos.numel():
                    ax = [pos[:, i].unique() for i in range(pos.shape[1])]
                    print("[bmmchk]   nan positions: %d of %d | dim0 uniq=%d %s"
                          " | dim1 uniq=%d %s | dim2 uniq=%d %s"
                          % (int(torch.isnan(o).sum()), o.numel(),
                             ax[0].numel(), ax[0][:4].tolist(),
                             ax[1].numel(), ax[1][:8].tolist(),
                             ax[2].numel(), ax[2][:8].tolist()),
                          file=_s.stderr, flush=True)
                # OUTPUT-SIDE TEST: hand the kernel a pre-poisoned y. Anything
                # still holding the sentinel was never written -- which would
                # mean the caller of the y=None path is seeing leftover heap.
                SENT = -7777.0
                yp = torch.full(o.shape, SENT, dtype=o.dtype, device=o.device)
                o4 = orig(a, w, wscale, y=yp, *[x for x in args if False], **{
                    k: v for k, v in kw.items() if k != "y"})
                o4 = o4[0] if isinstance(o4, (tuple, list)) else o4
                untouched = int((o4 == torch.tensor(SENT, dtype=o.dtype,
                                                    device=o.device)).sum())
                print("[bmmchk]   poisoned-y test: untouched=%d of %d, nan=%d"
                      % (untouched, o4.numel(), int(torch.isnan(o4).sum())),
                      file=_s.stderr, flush=True)
            except Exception as _e:
                print("[bmmchk]   rerun failed: %r" % (_e,), file=_s.stderr,
                      flush=True)
        return out

    mod.batched_gemm_a16wfp4 = chk
    print("[bmmchk] absorption BMM checker installed", file=_s.stderr, flush=True)


if _BMMCHK:
    _prev_am_bc = _HOOKS.get("atom.model_ops.attention_mla")

    def _am_with_bc(mod, _p=_prev_am_bc):
        if _p is not None:
            _p(mod)
        try:
            _install_bmmchk(mod)
        except Exception as exc:
            print("[bmmchk] hook failed: %r" % (exc,), file=sys.stderr, flush=True)

    _HOOKS["atom.model_ops.attention_mla"] = _am_with_bc


# --------------------------------------------------------------------------
# GLM_BMMDUMP=<path> -- capture the exact inputs of the first
# batched_gemm_a16wfp4 call that returns NaN from finite inputs, so the failure
# can be replayed without the 408 GB model.
# --------------------------------------------------------------------------
_BMMDUMP = os.environ.get("GLM_BMMDUMP")


def _install_bmmdump(mod):
    import sys as _s

    import torch

    if getattr(_install_bmmdump, "_done", False):
        return
    _install_bmmdump._done = True
    orig = mod.batched_gemm_a16wfp4
    st = {"n": 0, "saved": False}

    def dumper(a, w, wscale, y=None, transpose_bm=True, prequant=True,
               y_scale=None, *args, **kw):
        out = orig(a, w, wscale, y=y, transpose_bm=transpose_bm,
                   prequant=prequant, y_scale=y_scale, *args, **kw)
        st["n"] += 1
        if st["saved"] or a.shape[1] <= 2048:
            return out
        o = out[0] if isinstance(out, (tuple, list)) else out
        if not bool(torch.isnan(o).any()):
            return out
        if bool(torch.isnan(a.float()).any()):
            return out          # poison came from upstream, not this kernel
        st["saved"] = True
        import numpy as np
        np.savez(_BMMDUMP,
                 a=a.detach().float().cpu().numpy().astype(np.float32),
                 a_dtype=np.array(str(a.dtype)),
                 w=w.detach().cpu().numpy(),
                 wscale=wscale.detach().cpu().numpy(),
                 out=o.detach().float().cpu().numpy().astype(np.float32),
                 transpose_bm=np.array(transpose_bm),
                 prequant=np.array(prequant),
                 call=np.array(st["n"]))
        print("[bmmdump] saved failing call#%d -> %s  a%s w%s s%s"
              % (st["n"], _BMMDUMP, tuple(a.shape), tuple(w.shape),
                 tuple(wscale.shape)), file=_s.stderr, flush=True)
        return out

    mod.batched_gemm_a16wfp4 = dumper
    print("[bmmdump] installed", file=_s.stderr, flush=True)


if _BMMDUMP:
    _prev_am_bd = _HOOKS.get("atom.model_ops.attention_mla")

    def _am_with_bd(mod, _p=_prev_am_bd):
        if _p is not None:
            _p(mod)
        try:
            _install_bmmdump(mod)
        except Exception as exc:
            print("[bmmdump] hook failed: %r" % (exc,), file=sys.stderr, flush=True)

    _HOOKS["atom.model_ops.attention_mla"] = _am_with_bd


# --------------------------------------------------------------------------
# GLM_BMM_CONTIG=1 -- workaround for the layout-sensitive absorption BMM.
#
# In-model, batched_gemm_a16wfp4 returns exactly one NaN per output row (2568 of
# 84M) for GLM-5.2 layer 24. Re-running with the SAME tensors reproduces it
# exactly; re-running with .clone()d inputs is clean. Same values, same
# autotune config, same kernel -- so the fault tracks the memory layout, not the
# data. ATOM hands it a strided view (the 192-of-256 q_nope slice, transposed).
# Materialising that view sidesteps it.
# --------------------------------------------------------------------------
_BMMCONTIG = os.environ.get("GLM_BMM_CONTIG") == "1"


def _install_bmm_contig(mod):
    import sys as _s

    if getattr(_install_bmm_contig, "_done", False):
        return
    _install_bmm_contig._done = True
    orig = mod.batched_gemm_a16wfp4
    st = {"n": 0, "fixed": 0}

    def contig(a, w, wscale, *args, **kw):
        st["n"] += 1
        if not a.is_contiguous():
            a = a.contiguous()
            st["fixed"] += 1
        if not w.is_contiguous():
            w = w.contiguous()
        if not wscale.is_contiguous():
            wscale = wscale.contiguous()
        return orig(a, w, wscale, *args, **kw)

    mod.batched_gemm_a16wfp4 = contig

    import atexit
    atexit.register(lambda: print(
        "[bmmcontig] calls=%d, made contiguous=%d" % (st["n"], st["fixed"]),
        file=_s.stderr, flush=True))
    print("[bmmcontig] installed", file=_s.stderr, flush=True)


if _BMMCONTIG:
    _prev_am_ct = _HOOKS.get("atom.model_ops.attention_mla")

    def _am_with_ct(mod, _p=_prev_am_ct):
        if _p is not None:
            _p(mod)
        try:
            _install_bmm_contig(mod)
        except Exception as exc:
            print("[bmmcontig] hook failed: %r" % (exc,), file=sys.stderr, flush=True)

    _HOOKS["atom.model_ops.attention_mla"] = _am_with_ct


# --------------------------------------------------------------------------
# GLM_BMM_OWNY=1 -- workaround: allocate the BMM output ourselves.
#
# batched_gemm_a16wfp4 with y=None leaves the LAST element of each row's
# (B-1, N-1) corner holding heap garbage -- out[:, 63, 511] was NaN for all 2568
# rows at GLM-5.2 layer 24, deterministically. Handing the kernel an explicit y
# makes the same call write every element and produce no NaN (measured:
# untouched=0, nan=0). Allocate with zeros so that even if the corner is skipped
# the value is benign instead of whatever the allocator had.
# --------------------------------------------------------------------------
_BMMOWNY = os.environ.get("GLM_BMM_OWNY") == "1"


def _install_bmm_owny(mod):
    import sys as _s

    import torch

    if getattr(_install_bmm_owny, "_done", False):
        return
    _install_bmm_owny._done = True
    orig = mod.batched_gemm_a16wfp4
    st = {"n": 0, "owned": 0}

    def owny(a, w, wscale, y=None, transpose_bm=True, prequant=True,
             y_scale=None, *args, **kw):
        st["n"] += 1
        if y is None:
            B, M, _ = a.shape
            N = w.shape[1]
            shape = (M, B, N) if transpose_bm else (B, M, N)
            y = torch.zeros(shape, dtype=a.dtype, device=a.device)
            st["owned"] += 1
        return orig(a, w, wscale, y=y, transpose_bm=transpose_bm,
                    prequant=prequant, y_scale=y_scale, *args, **kw)

    mod.batched_gemm_a16wfp4 = owny

    import atexit
    atexit.register(lambda: print(
        "[bmmowny] calls=%d, output allocated by us=%d" % (st["n"], st["owned"]),
        file=_s.stderr, flush=True))
    print("[bmmowny] installed", file=_s.stderr, flush=True)


if _BMMOWNY:
    _prev_am_oy = _HOOKS.get("atom.model_ops.attention_mla")

    def _am_with_oy(mod, _p=_prev_am_oy):
        if _p is not None:
            _p(mod)
        try:
            _install_bmm_owny(mod)
        except Exception as exc:
            print("[bmmowny] hook failed: %r" % (exc,), file=sys.stderr, flush=True)

    _HOOKS["atom.model_ops.attention_mla"] = _am_with_oy


# --------------------------------------------------------------------------
# GLM_PCTAP=<path> -- prefix-caching divergence tap.  For each prefill forward,
# save the last row's residual stream at every decoder layer, and the logits of
# the first decode step.  Unlike GLM_ATTN0 this does NOT skip short forwards:
# with prefix caching on, the suffix prefill is exactly the short forward we
# need to see.
# --------------------------------------------------------------------------
_PCTAP = os.environ.get("GLM_PCTAP")
_pctap_logits = True


def _install_pctap(torch_mod, path):
    import numpy as _np
    torch = torch_mod
    st = {"L": 0, "fwd": 0, "rows": [], "logit_n": 0}
    store = {}

    def hook(module, args, output):
        # GUARDS FIRST. A .cpu() copy during CUDA-graph capture raises
        # "Cannot copy between CPU and CUDA tensors during CUDA graph capture",
        # which kills engine init -- so nothing may touch a tensor above this.
        try:
            from atom.utils.forward_context import get_forward_context
            if get_forward_context().context.is_dummy_run:
                return
        except Exception:
            pass
        try:
            if torch.cuda.is_current_stream_capturing():
                return
        except Exception:
            pass
        # Logits: any module emitting a vocab-sized last dim. Caught here rather
        # than by wrapping aiter's sampler at install time, because that scan
        # ran before aiter was imported and matched nothing.
        _o = output[0] if isinstance(output, (tuple, list)) else output
        if hasattr(_o, "detach") and _o.dim() >= 2 and _o.shape[-1] > 100000:
            if st["logit_n"] < 24:
                store["logits_f%d_n%d" % (st["fwd"], st["logit_n"])] = \
                    _o[-1].detach().float().cpu().numpy()
                st["logit_n"] += 1
                try:
                    _np.savez(path, rows=_np.array(st["rows"]), **store)
                except Exception:
                    pass
            return
        if type(module).__name__ != "DeepseekV2DecoderLayer":
            return
        x = output[0] if isinstance(output, (tuple, list)) else output
        if not hasattr(x, "detach") or x.dim() < 2:
            return
        n = x.shape[0]
        if n < 2:                      # decode step: not a prefill
            return
        L = st["L"]
        key = "f%d_l%d" % (st["fwd"], L)
        store[key + "_hid"] = x[-1].detach().float().cpu().numpy()
        if isinstance(output, (tuple, list)) and len(output) > 1 and \
                hasattr(output[1], "detach"):
            store[key + "_res"] = output[1][-1].detach().float().cpu().numpy()
        st["L"] += 1
        if st["L"] >= 78:
            st["L"] = 0
            st["rows"].append(n)
            st["fwd"] += 1
            # Flush now: the model runs in the TP0 subprocess and atexit there
            # is not guaranteed, so an end-of-run save loses everything.
            try:
                _np.savez(path, rows=_np.array(st["rows"]), **store)
                print("[pctap] fwd%d rows=%d flushed (%d arrays)"
                      % (st["fwd"] - 1, n, len(store)),
                      file=__import__("sys").stderr, flush=True)
            except Exception as _e:
                print("[pctap] flush failed %r" % (_e,),
                      file=__import__("sys").stderr, flush=True)

    torch_mod.nn.modules.module.register_module_forward_hook(hook)

    # logits of the first few sampled tokens
    import sys as _s
    for mn, m in list(_s.modules.items()):
        if m is None or (mn != "aiter" and not mn.startswith("aiter.")):
            continue
        for op in ("greedy_sample", "mixed_sample_outer_exponential"):
            f = getattr(m, op, None)
            if f is None or getattr(f, "_pctap", False):
                continue

            def wrap(orig, name=op):
                def w(out, input, *a, **kw):
                    r = orig(out, input, *a, **kw)
                    if st["logit_n"] < 8 and hasattr(input, "detach"):
                        store["logit%d" % st["logit_n"]] = \
                            input[0].detach().float().cpu().numpy()
                        st["logit_n"] += 1
                        try:
                            _np.savez(path, rows=_np.array(st["rows"]), **store)
                        except Exception:
                            pass
                    return r
                w._pctap = True
                return w
            setattr(m, op, wrap(f))

    import atexit

    def _save():
        if not store:
            return          # parent process: never clobber the subprocess file
        store["rows"] = _np.array(st["rows"])
        _np.savez(path, **store)
        print("[pctap] %d arrays -> %s.npz  rows=%s"
              % (len(store), path, st["rows"][:12]), file=_s.stderr, flush=True)

    atexit.register(_save)
    print("[pctap] installed -> %s" % path, file=_s.stderr, flush=True)


if _PCTAP:
    _prev_torch_pctap = _HOOKS.get("torch")

    def _patch_torch_with_pctap(mod, _p=_prev_torch_pctap):
        if _p is not None:
            _p(mod)
        try:
            _install_pctap(mod, _PCTAP)
        except Exception as exc:
            print("[pctap] failed: %r" % (exc,), file=sys.stderr, flush=True)

    _HOOKS["torch"] = _patch_torch_with_pctap


# --------------------------------------------------------------------------
# GLM_ROUTETAP=<path> -- capture MoE expert selection for the last row of each
# prefill forward, per layer. Answers: does chunked prefill change WHICH
# experts a token routes to?
# --------------------------------------------------------------------------
_ROUTETAP = os.environ.get("GLM_ROUTETAP")


def _install_routetap(mod, path):
    """Wrap aiter's moe_routing.topk, which returns (expt_scal, expt_indx,
    bitmatrix) -- expt_indx is the per-token expert selection we need.
    RoutingData only carries the histogram, not the per-token indices."""
    import sys as _s
    import numpy as _np
    import importlib

    R = mod
    orig = getattr(R, "topk", None)
    if orig is None or getattr(orig, "_routetap", False):
        return
    st = {"n": 0}
    store = {}

    def wrapped(x, *a, **kw):
        r = orig(x, *a, **kw)
        try:
            if hasattr(x, "detach") and x.dim() == 2 and x.shape[0] > 2:
                scal, indx = r[0], r[1]
                k = "c%04d" % st["n"]
                store[k + "_rows"] = _np.array([x.shape[0]])
                store[k + "_logit"] = x[-1].detach().float().cpu().numpy()
                store[k + "_expt"] = indx[-1].detach().cpu().numpy()
                store[k + "_scal"] = scal[-1].detach().float().cpu().numpy()
                st["n"] += 1
                if st["n"] % 25 == 0:
                    _np.savez(path, **store)
        except Exception as exc:
            print("[routetap] %r" % (exc,), file=_s.stderr, flush=True)
        return r

    wrapped._routetap = True
    R.topk = wrapped
    print("[routetap] wrapped moe_routing.topk", file=_s.stderr, flush=True)

    import atexit

    def _save():
        if store:
            _np.savez(path, **store)
            print("[routetap] %d calls -> %s.npz" % (st["n"], path),
                  file=_s.stderr, flush=True)
    atexit.register(_save)


if _ROUTETAP:
    # Key on the module that defines routing(), so the hook fires exactly when
    # it is imported. Keying on ATOM's importer never dispatched.
    _MODK = "aiter.ops.triton.moe.moe_routing.routing"
    _prev_fmt = _HOOKS.get(_MODK)

    def _fmt_with_routetap(mod, _p=_prev_fmt):
        if _p is not None:
            _p(mod)
        try:
            _install_routetap(mod, _ROUTETAP)
        except Exception as exc:
            print("[routetap] hook failed: %r" % (exc,), file=sys.stderr, flush=True)

    _HOOKS[_MODK] = _fmt_with_routetap


# --------------------------------------------------------------------------
# GLM_MOETAP=<path> -- MoE block input/output for the last prompt token, for
# the first GLM_MOETAP_LAYERS MoE layers. Pairs with the router dump that
# aiter's routing.py writes under the same env var.
# --------------------------------------------------------------------------
_MOETAP = os.environ.get("GLM_MOETAP")
_MOETAP_L = int(os.environ.get("GLM_MOETAP_LAYERS", "6"))


def _install_moetap(torch_mod, path):
    import sys as _s
    import numpy as _np
    torch = torch_mod
    st = {"dec": 0, "moe": 0, "fwd": 0}
    store = {}

    def hook(module, args, output):
        # Guards first: a .cpu() copy during CUDA-graph capture is fatal.
        try:
            from atom.utils.forward_context import get_forward_context
            if get_forward_context().context.is_dummy_run:
                return
        except Exception:
            pass
        try:
            if torch.cuda.is_current_stream_capturing():
                return
        except Exception:
            pass
        n = type(module).__name__
        if n == "DeepseekV2MoE":
            x = args[0] if args else None
            y = output[0] if isinstance(output, (tuple, list)) else output
            if (hasattr(x, "detach") and hasattr(y, "detach")
                    and x.dim() >= 2 and x.shape[0] >= 2
                    and st["moe"] < _MOETAP_L):
                k = "f%d_moe%d" % (st["fwd"], st["moe"])
                store[k + "_in"] = x[-1].detach().float().cpu().numpy()
                store[k + "_out"] = y[-1].detach().float().cpu().numpy()
                store[k + "_rows"] = _np.array([x.shape[0]])
            if hasattr(args[0] if args else None, "shape") and args[0].shape[0] >= 2:
                st["moe"] += 1
            return
        if n != "DeepseekV2DecoderLayer":
            return
        x = output[0] if isinstance(output, (tuple, list)) else output
        if not hasattr(x, "detach") or x.dim() < 2 or x.shape[0] < 2:
            return
        st["dec"] += 1
        if st["dec"] >= 78:
            st["dec"] = 0
            st["moe"] = 0
            st["fwd"] += 1
            try:
                _np.savez(path + ".moe", **store)
            except Exception:
                pass

    torch_mod.nn.modules.module.register_module_forward_hook(hook)
    print("[moetap] module hook installed -> %s.moe.npz" % path,
          file=_s.stderr, flush=True)


if _MOETAP:
    _prev_torch_moetap = _HOOKS.get("torch")

    def _patch_torch_with_moetap(mod, _p=_prev_torch_moetap):
        if _p is not None:
            _p(mod)
        try:
            _install_moetap(mod, _MOETAP)
        except Exception as exc:
            print("[moetap] failed: %r" % (exc,), file=sys.stderr, flush=True)

    _HOOKS["torch"] = _patch_torch_with_moetap


# --------------------------------------------------------------------------
# GLM_MOEREPLAY=1 -- batch-composition dependence of the fused MoE block.
# --------------------------------------------------------------------------
_MOEREPLAY = os.environ.get("GLM_MOEREPLAY") == "1"
_MOEREPLAY_N = int(os.environ.get("GLM_MOEREPLAY_LAYERS", "3"))


def _install_moereplay(torch_mod):
    import sys as _s
    torch = torch_mod
    st = {"n": 0, "busy": False}

    def hook(module, args, output):
        if type(module).__name__ != "DeepseekV2MoE":
            return
        if st["busy"] or st["n"] >= _MOEREPLAY_N:
            return
        try:
            from atom.utils.forward_context import get_forward_context
            if get_forward_context().context.is_dummy_run:
                return
        except Exception:
            pass
        try:
            if torch.cuda.is_current_stream_capturing():
                return
        except Exception:
            pass
        x = args[0] if args else None
        y = output[0] if isinstance(output, (tuple, list)) else output
        if not (hasattr(x, "detach") and hasattr(y, "detach")):
            return
        if x.dim() < 2 or x.shape[0] < 512:
            return

        N = x.shape[0]
        ref = y[-1].detach().float()
        st["busy"] = True
        try:
            print("[moereplay] MoE#%d  batch N=%d  ||out[-1]||=%.4f"
                  % (st["n"], N, float(ref.norm())), file=_s.stderr, flush=True)
            for T in (N, N // 2, 262, 128, 64, 16, 1):
                if T > N or T < 1:
                    continue
                with torch.inference_mode():
                    y2 = module(x[-T:].contiguous())
                y2 = y2[0] if isinstance(y2, (tuple, list)) else y2
                cur = y2[-1].detach().float()
                md = float((cur - ref).abs().max())
                rel = float((cur - ref).norm() / (ref.norm() + 1e-30))
                tag = "  <== CONTROL (must be 0)" if T == N else ""
                print("[moereplay]    T=%5d  |max d|=%.4e  rel=%.4e%s"
                      % (T, md, rel, tag), file=_s.stderr, flush=True)
        except Exception as exc:
            print("[moereplay] failed: %r" % (exc,), file=_s.stderr, flush=True)
        finally:
            st["busy"] = False
            st["n"] += 1

    torch_mod.nn.modules.module.register_module_forward_hook(hook)
    print("[moereplay] installed", file=_s.stderr, flush=True)


if _MOEREPLAY:
    _prev_torch_mr = _HOOKS.get("torch")

    def _patch_torch_with_moereplay(mod, _p=_prev_torch_mr):
        if _p is not None:
            _p(mod)
        try:
            _install_moereplay(mod)
        except Exception as exc:
            print("[moereplay] hook failed: %r" % (exc,), file=sys.stderr, flush=True)

    _HOOKS["torch"] = _patch_torch_with_moereplay


# --------------------------------------------------------------------------
# GLM_DECTAP=<path> -- per-row LM-head logits, decode steps included.
# --------------------------------------------------------------------------
_DECTAP = os.environ.get("GLM_DECTAP")
_DECTAP_N = int(os.environ.get("GLM_DECTAP_CAPS", "80"))
_DECTAP_K = int(os.environ.get("GLM_DECTAP_TOPK", "8"))


def _install_dectap(torch_mod, path):
    import sys as _s
    import numpy as _np
    torch = torch_mod
    st = {"n": 0}
    store = {}

    def hook(module, args, output):
        # Guards first: a .cpu() copy during CUDA-graph capture is fatal.
        try:
            from atom.utils.forward_context import get_forward_context
            if get_forward_context().context.is_dummy_run:
                return
        except Exception:
            pass
        try:
            if torch.cuda.is_current_stream_capturing():
                return
        except Exception:
            pass
        o = output[0] if isinstance(output, (tuple, list)) else output
        if not (hasattr(o, "detach") and o.dim() >= 2 and o.shape[-1] > 100000):
            return
        if st["n"] >= _DECTAP_N:
            return
        try:
            f = o.detach().float()
            v, i = torch.topk(f, _DECTAP_K, dim=-1)
            k = "c%03d" % st["n"]
            store[k + "_rows"] = _np.array([f.shape[0]])
            store[k + "_ids"] = i.cpu().numpy()
            store[k + "_val"] = v.cpu().numpy()
            st["n"] += 1
            if st["n"] % 5 == 0 or st["n"] < 6:
                _np.savez(path + "_%d" % os.getpid(), **store)
        except Exception as exc:
            print("[dectap] %r" % (exc,), file=_s.stderr, flush=True)

    torch_mod.nn.modules.module.register_module_forward_hook(hook)
    print("[dectap] installed -> %s.npz" % path, file=_s.stderr, flush=True)

    import atexit

    def _save():
        if store:
            _np.savez(path + "_%d" % os.getpid(), **store)
            print("[dectap] %d captures saved (pid %d)" % (st["n"], os.getpid()),
                  file=_s.stderr, flush=True)
    atexit.register(_save)


if _DECTAP:
    _prev_torch_dt = _HOOKS.get("torch")

    def _patch_torch_with_dectap(mod, _p=_prev_torch_dt):
        if _p is not None:
            _p(mod)
        try:
            _install_dectap(mod, _DECTAP)
        except Exception as exc:
            print("[dectap] hook failed: %r" % (exc,), file=sys.stderr, flush=True)

    _HOOKS["torch"] = _patch_torch_with_dectap


# --------------------------------------------------------------------------
# GLM_DECLAYER=<path> -- decode-step residual stream, every layer, every row.
# --------------------------------------------------------------------------
_DECLAYER = os.environ.get("GLM_DECLAYER")
_DECLAYER_STEPS = int(os.environ.get("GLM_DECLAYER_STEPS", "8"))
_DECLAYER_SKIP = int(os.environ.get("GLM_DECLAYER_SKIP", "0"))


def _install_declayer(torch_mod, path):
    import sys as _s
    import numpy as _np
    torch = torch_mod
    st = {"L": 0, "step": 0}
    store = {}

    def flush():
        try:
            _np.savez(path + "_%d" % os.getpid(), **store)
        except Exception:
            pass

    def hook(module, args, output):
        # Guards first: a .cpu() copy during CUDA-graph capture is fatal.
        try:
            from atom.utils.forward_context import get_forward_context
            if get_forward_context().context.is_dummy_run:
                return
        except Exception:
            pass
        try:
            if torch.cuda.is_current_stream_capturing():
                return
        except Exception:
            pass
        if st["step"] >= _DECLAYER_SKIP + _DECLAYER_STEPS:
            return
        n = type(module).__name__

        # LM head: vocab-sized last dim -> label this step and advance
        o = output[0] if isinstance(output, (tuple, list)) else output
        if hasattr(o, "detach") and o.dim() >= 2 and o.shape[-1] > 100000:
            f = o.detach().float()
            v, i = torch.topk(f, 8, dim=-1)
            k = "s%02d" % st["step"]
            store[k + "_top_ids"] = i.cpu().numpy()
            store[k + "_top_val"] = v.cpu().numpy()
            store[k + "_nrows"] = _np.array([f.shape[0]])
            st["step"] += 1
            st["L"] = 0
            flush()
            return

        if n != "DeepseekV2DecoderLayer":
            return
        x = output[0] if isinstance(output, (tuple, list)) else output
        if not hasattr(x, "detach") or x.dim() < 2:
            return
        # DECODE only: one row per active sequence, so a small row count.
        if x.shape[0] > 8:
            return
        if st["step"] < _DECLAYER_SKIP:
            st["L"] += 1
            return
        k = "s%02d_l%02d" % (st["step"], st["L"])
        store[k + "_hid"] = x.detach().float().cpu().numpy()
        if isinstance(output, (tuple, list)) and len(output) > 1 and \
                hasattr(output[1], "detach"):
            store[k + "_res"] = output[1].detach().float().cpu().numpy()
        st["L"] += 1

    torch_mod.nn.modules.module.register_module_forward_hook(hook)
    print("[declayer] installed -> %s_<pid>.npz (%d decode steps)"
          % (path, _DECLAYER_STEPS), file=_s.stderr, flush=True)

    import atexit

    def _save():
        if store:
            flush()
            print("[declayer] %d steps captured (pid %d)" % (st["step"], os.getpid()),
                  file=_s.stderr, flush=True)
    atexit.register(_save)


if _DECLAYER:
    _prev_torch_dl = _HOOKS.get("torch")

    def _patch_torch_with_declayer(mod, _p=_prev_torch_dl):
        if _p is not None:
            _p(mod)
        try:
            _install_declayer(mod, _DECLAYER)
        except Exception as exc:
            print("[declayer] hook failed: %r" % (exc,), file=sys.stderr, flush=True)

    _HOOKS["torch"] = _patch_torch_with_declayer


# --------------------------------------------------------------------------
# GLM_LAYERTAP=<path> -- sub-module in/out for one decoder layer, decode steps.
# --------------------------------------------------------------------------
_LAYERTAP = os.environ.get("GLM_LAYERTAP")
_LAYERTAP_L = int(os.environ.get("GLM_LAYERTAP_LAYER", "58"))
_LAYERTAP_SKIP = int(os.environ.get("GLM_LAYERTAP_SKIP", "0"))
_LAYERTAP_STEPS = int(os.environ.get("GLM_LAYERTAP_STEPS", "8"))


def _install_layertap(torch_mod, path):
    import sys as _s
    import numpy as _np
    torch = torch_mod
    st = {"L": 0, "step": 0, "sub": 0}
    store = {}

    def flush():
        try:
            _np.savez(path + "_%d" % os.getpid(), **store)
        except Exception:
            pass

    def hook(module, args, output):
        try:
            from atom.utils.forward_context import get_forward_context
            if get_forward_context().context.is_dummy_run:
                return
        except Exception:
            pass
        try:
            if torch.cuda.is_current_stream_capturing():
                return
        except Exception:
            pass
        if st["step"] >= _LAYERTAP_SKIP + _LAYERTAP_STEPS:
            return
        cls = type(module).__name__

        o = output[0] if isinstance(output, (tuple, list)) else output

        # LM head marks the end of a step
        if hasattr(o, "detach") and o.dim() >= 2 and o.shape[-1] > 100000:
            f = o.detach().float()
            v, i = torch.topk(f, 8, dim=-1)
            k = "s%02d" % st["step"]
            store[k + "_top_ids"] = i.cpu().numpy()
            store[k + "_top_val"] = v.cpu().numpy()
            st["step"] += 1
            st["L"] = 0
            st["sub"] = 0
            flush()
            return

        if cls == "DeepseekV2DecoderLayer":
            st["L"] += 1
            st["sub"] = 0
            return

        if st["L"] != _LAYERTAP_L or st["step"] < _LAYERTAP_SKIP:
            return
        if not hasattr(o, "detach") or o.dim() < 2 or o.shape[0] > 8:
            return

        k = "s%02d_%03d_%s" % (st["step"], st["sub"], cls)
        try:
            store[k + "_out"] = o.detach().float().cpu().numpy()
            a = args[0] if args else None
            if hasattr(a, "detach") and a.dim() >= 2 and a.shape[0] <= 8:
                store[k + "_in"] = a.detach().float().cpu().numpy()
            st["sub"] += 1
        except Exception as exc:
            print("[layertap] %r" % (exc,), file=_s.stderr, flush=True)

    torch_mod.nn.modules.module.register_module_forward_hook(hook)
    print("[layertap] installed layer=%d window=[%d,%d) -> %s_<pid>.npz"
          % (_LAYERTAP_L, _LAYERTAP_SKIP, _LAYERTAP_SKIP + _LAYERTAP_STEPS, path),
          file=_s.stderr, flush=True)

    import atexit

    def _save():
        if store:
            flush()
            print("[layertap] %d steps, %d arrays (pid %d)"
                  % (st["step"], len(store), os.getpid()), file=_s.stderr, flush=True)
    atexit.register(_save)


if _LAYERTAP:
    _prev_torch_lt = _HOOKS.get("torch")

    def _patch_torch_with_layertap(mod, _p=_prev_torch_lt):
        if _p is not None:
            _p(mod)
        try:
            _install_layertap(mod, _LAYERTAP)
        except Exception as exc:
            print("[layertap] hook failed: %r" % (exc,), file=sys.stderr, flush=True)

    _HOOKS["torch"] = _patch_torch_with_layertap



# --------------------------------------------------------------------------
# GLM_ATTNTAP=<path> -- arguments and output of the decode attention kernel.
# Counter rides the torch hook; the wrap rides the AITER hook (aiter is not
# imported yet when the torch hook runs).
# --------------------------------------------------------------------------
_ATTNTAP = os.environ.get("GLM_ATTNTAP")
_ATTNTAP_L = int(os.environ.get("GLM_ATTNTAP_LAYER", "58"))
_ATTNTAP_SKIP = int(os.environ.get("GLM_ATTNTAP_SKIP", "0"))
_ATTNTAP_STEPS = int(os.environ.get("GLM_ATTNTAP_STEPS", "8"))
_ATTNTAP_MAXNUMEL = 5_000_000
_ATTNTAP_KV = True          # never copy the paged KV cache

_AT = {"L": 0, "step": 0, "store": {}, "path": None, "hits": {}}


def _at_flush():
    import numpy as _np
    if _AT["path"] and _AT["store"]:
        try:
            _np.savez(_AT["path"] + "_%d" % os.getpid(), **_AT["store"])
        except Exception:
            pass


def _install_attntap_counter(torch_mod, path):
    import sys as _s
    torch = torch_mod
    _AT["path"] = path

    def hook(module, args, output):
        try:
            from atom.utils.forward_context import get_forward_context
            if get_forward_context().context.is_dummy_run:
                return
        except Exception:
            pass
        try:
            if torch.cuda.is_current_stream_capturing():
                return
        except Exception:
            pass
        o = output[0] if isinstance(output, (tuple, list)) else output
        if hasattr(o, "detach") and o.dim() >= 2 and o.shape[-1] > 100000:
            f = o.detach().float()
            v, i = torch.topk(f, 8, dim=-1)
            k = "s%02d" % _AT["step"]
            _AT["store"][k + "_top_ids"] = i.cpu().numpy()
            _AT["store"][k + "_top_val"] = v.cpu().numpy()
            _AT["step"] += 1
            _AT["L"] = 0
            _at_flush()
            return
        if type(module).__name__ == "DeepseekV2DecoderLayer":
            _AT["L"] += 1

    torch_mod.nn.modules.module.register_module_forward_hook(hook)
    import atexit

    def _rep():
        _at_flush()
        print("[attntap] %d arrays, entry hits=%s (pid %d)"
              % (len(_AT["store"]), _AT["hits"], os.getpid()),
              file=_s.stderr, flush=True)
    atexit.register(_rep)


def _install_attntap_wrap(aiter_mod):
    import sys as _s
    import numpy as _np_at
    NAMES = ("mla_decode_fwd", "triton_shuffle_mla_decode_fwd",
             "decode_attention_fwd")
    total = 0
    for mn, m in list(_s.modules.items()):
        if m is None or (mn != "aiter" and not mn.startswith("aiter.")):
            continue
        for nm in NAMES:
            fn = getattr(m, nm, None)
            if fn is None or getattr(fn, "_attntap", False):
                continue

            def mk(orig, name=nm):
                def w(*a, **kw):
                    grab = (_AT["L"] == _ATTNTAP_L
                            and _ATTNTAP_SKIP <= _AT["step"]
                            < _ATTNTAP_SKIP + _ATTNTAP_STEPS)
                    if grab:
                        _AT["hits"][name] = _AT["hits"].get(name, 0) + 1
                        pre = "s%02d_%s" % (_AT["step"], name)
                        try:
                            for j, t in enumerate(a):
                                if hasattr(t, "detach") and hasattr(t, "numel") \
                                        and t.numel() <= _ATTNTAP_MAXNUMEL:
                                    _AT["store"]["%s_arg%02d" % (pre, j)] = \
                                        t.detach().float().cpu().numpy() \
                                        if t.is_floating_point() \
                                        else t.detach().cpu().numpy()
                        except Exception as exc:
                            print("[attntap] pre %r" % (exc,),
                                  file=_s.stderr, flush=True)
                    if grab:
                        # gather the KV content this call will read, in the
                        # order the index list gives, for every row.
                        try:
                            import torch as _th
                            kvb = None
                            for t in a:
                                if hasattr(t, "numel") and t.numel() > _ATTNTAP_MAXNUMEL:
                                    kvb = t; break
                            idx = a[5] if len(a) > 5 else None
                            lens = a[6] if len(a) > 6 else None
                            if kvb is not None and idx is not None and lens is not None:
                                flat = kvb.reshape(-1, kvb.shape[-1])
                                nrow = idx.shape[0]
                                for rr in range(nrow):
                                    L = int(lens[rr])
                                    sl = idx[rr][:L].to(_th.int64)
                                    _AT["store"]["%s_kv_r%d" % (pre, rr)] = \
                                        flat[sl].detach().float().cpu().numpy()
                                    _AT["store"]["%s_kvidx_r%d" % (pre, rr)] = \
                                        sl.detach().cpu().numpy()
                                _AT["store"][pre + "_kvshape"] = \
                                    _np_at.array(list(kvb.shape))
                        except Exception as exc:
                            print("[attntap] kv %r" % (exc,),
                                  file=_s.stderr, flush=True)
                    r = orig(*a, **kw)
                    if grab:
                        try:
                            # the output tensor is an in-place arg in these APIs
                            for j, t in enumerate(a):
                                if hasattr(t, "detach") and hasattr(t, "numel") \
                                        and t.numel() <= _ATTNTAP_MAXNUMEL \
                                        and t.is_floating_point():
                                    _AT["store"]["%s_post%02d" % (pre, j)] = \
                                        t.detach().float().cpu().numpy()
                            _at_flush()
                        except Exception as exc:
                            print("[attntap] post %r" % (exc,),
                                  file=_s.stderr, flush=True)
                    return r
                w._attntap = True
                return w

            setattr(m, nm, mk(fn))
            total += 1
    print("[attntap] wrapped %d decode entry point(s) across aiter modules"
          % total, file=_s.stderr, flush=True)


if _ATTNTAP:
    _prev_torch_at = _HOOKS.get("torch")

    def _patch_torch_with_attntap(mod, _p=_prev_torch_at):
        if _p is not None:
            _p(mod)
        try:
            _install_attntap_counter(mod, _ATTNTAP)
        except Exception as exc:
            print("[attntap] counter failed: %r" % (exc,), file=sys.stderr, flush=True)

    _HOOKS["torch"] = _patch_torch_with_attntap

    _prev_aiter_at = _HOOKS.get("aiter")

    def _patch_aiter_with_attntap(mod, _p=_prev_aiter_at):
        if _p is not None:
            _p(mod)
        try:
            _install_attntap_wrap(mod)
        except Exception as exc:
            print("[attntap] wrap failed: %r" % (exc,), file=sys.stderr, flush=True)

    _HOOKS["aiter"] = _patch_aiter_with_attntap




# --------------------------------------------------------------------------
# GLM_KVW=<path> -- KV-write inputs + layer hidden states during prefill.
# GLM_KVW_LAYER=-1 -> every layer;  GLM_KVW_SLOT=<n> -> only that token's row.
# --------------------------------------------------------------------------
_KVW = os.environ.get("GLM_KVW")
_KVW_L = int(os.environ.get("GLM_KVW_LAYER", "58"))
_KVW_SLOT = int(os.environ.get("GLM_KVW_SLOT", "-1"))
_KVW_SUB = int(os.environ.get("GLM_KVW_SUB", "-1"))
_KVW_MINTOK = int(os.environ.get("GLM_KVW_MINTOK", "64"))   # prefill only
_KVW_BUDGET = 400_000_000
_KVW_PATHS = {}
_KVW_STACK = []
_KVW_ATT = int(os.environ.get("GLM_KVW_ATT", "-1"))
_KVW_LOGITS = int(os.environ.get("GLM_KVW_LOGITS", "-1"))
_MQA_IN = os.environ.get("GLM_MQA_IN") == "1"
_MQA_CU = os.environ.get("GLM_MQA_CU") == "1"                                   # bytes, hard cap

_KW = {"L": 0, "fwd": 0, "store": {}, "path": None, "hits": {},
       "wrapped": 0, "bytes": 0, "done": False, "row": None, "rowfwd": None,
       "sub": 0}


def _kw_flush():
    """Serialise once. Called from atexit only -- never per-call."""
    import numpy as _np
    if _KW["path"] and _KW["store"]:
        try:
            _np.savez(_KW["path"] + "_%d" % os.getpid(), **_KW["store"])
        except Exception as exc:
            import sys as _s
            print("[kvw] flush failed: %r" % (exc,), file=_s.stderr, flush=True)


def _kw_put(key, t):
    if _KW["bytes"] > _KVW_BUDGET:
        return
    v = t.detach()
    a = v.float().cpu().numpy() if v.is_floating_point() else v.cpu().numpy()
    _KW["store"][key] = a
    _KW["bytes"] += a.nbytes


_KVW_ATT_NAMES = ("mla_decode_fwd", "mla_prefill_fwd")


def _kvw_att_grab(name, a, kw):
    """Dump the attention call's metadata (never the paged cache itself)."""
    pre = "f%d_ATT_%s" % (_KW["fwd"], name)
    _KW["hits"]["ATT:" + name] = _KW["hits"].get("ATT:" + name, 0) + 1
    items = [("a%02d" % j, t) for j, t in enumerate(a)]
    items += [(k, kw[k]) for k in sorted(kw)]
    for nm, t in items:
        if t is None or not hasattr(t, "numel"):
            continue
        # metadata is small; the KV cache is orders of magnitude larger
        if 0 < t.numel() <= 20_000_000:
            _kw_put("%s_%s" % (pre, nm), t)


_KVW_NAMES = (
    "triton_cat_and_cache_mla",
    "triton_fused_qk_rope_cat_and_cache_mla",
    "concat_and_cache_mla",
    "concat_and_cache_mla_seg",
    "fused_qk_rope_concat_and_cache_mla",
    "fused_qk_rope_concat_and_cache_mla_seg",
)


def _kvw_try_wrap():
    """Wrap the KV-write entry points where the call sites resolve them."""
    import sys as _s
    m = _s.modules.get("atom.model_ops.attention_mla")
    if m is None:
        return
    n = 0
    for nm in _KVW_NAMES:
        fn = getattr(m, nm, None)
        if fn is None or getattr(fn, "_kvw", False):
            continue

        def mk(orig, name=nm):
            def w(*a, **kw):
                try:
                    if _KVW_L < 0 or _KW["L"] == _KVW_L:
                        _kvw_grab(name, a, kw)
                except Exception as exc:
                    print("[kvw] grab %s: %r" % (name, exc),
                          file=_s.stderr, flush=True)
                return orig(*a, **kw)
            w._kvw = True
            return w

        setattr(m, nm, mk(fn))
        n += 1
    if _KVW_LOGITS >= 0:
        lm = None
        for _mn in ("atom.models.deepseek_v2",
                    "atom.plugin.vllm.attention.layer_sparse_mla"):
            _c = _s.modules.get(_mn)
            if _c is not None and getattr(_c, "fp8_mqa_logits", None) is not None:
                lm = _c
                break
        f0 = getattr(lm, "fp8_mqa_logits", None) if lm is not None else None
        if f0 is not None and not getattr(f0, "_kvwl", False):
            def mkl(orig):
                def wl(*a, **kw):
                    r = orig(*a, **kw)
                    try:
                        n = int(r.shape[0])
                        if n > _KVW_LOGITS:
                            k = "LOGITS_L%02d_n%d_r%d" % (_KW["L"], n, _KVW_LOGITS)
                            if k not in _KW["store"]:
                                _kw_put(k, r[_KVW_LOGITS])
                                if _MQA_IN:
                                    _mqa_grab_inputs(a, kw, n)
                                _KW["hits"]["logits_n%d" % n] = \
                                    _KW["hits"].get("logits_n%d" % n, 0) + 1
                        if _MQA_CU:
                            _mqa_grab_cu(a, kw, n)
                    except Exception as exc:
                        print("[kvw] logits %r" % (exc,),
                              file=_s.stderr, flush=True)
                    return r
                wl._kvwl = True
                return wl

            setattr(lm, "fp8_mqa_logits", mkl(f0))
            print("[kvw] indexer-logits tap on row %d" % _KVW_LOGITS,
                  file=_s.stderr, flush=True)
    if _KVW_ATT >= 0:
        for nm in _KVW_ATT_NAMES:
            fn = getattr(m, nm, None)
            if fn is None or getattr(fn, "_kvwa", False):
                continue

            def mka(orig, name=nm):
                def wa(*a, **kw):
                    try:
                        q = a[0] if a else None
                        if _KW["L"] == _KVW_ATT and q is not None \
                                and hasattr(q, "shape") and q.shape[0] > 100:
                            _kvw_att_grab(name, a, kw)
                    except Exception as exc:
                        print("[kvw] att %r" % (exc,), file=_s.stderr, flush=True)
                    return orig(*a, **kw)
                wa._kvwa = True
                return wa

            setattr(m, nm, mka(fn))
        print("[kvw] attention metadata tap on layer %d" % _KVW_ATT,
              file=_s.stderr, flush=True)
    # count the prefill implementations too, keyed by layer
    try:
        import sys as _s2
        am = _s2.modules.get("atom.model_ops.attention_mla")
        for cn in ("DeepseekV2MLAAttentionImpl", "MLAAttentionImpl", "MLAImpl"):
            cls = getattr(am, cn, None)
            if cls is not None:
                break
        else:
            cls = None
        if cls is None:
            for _o in vars(am).values():
                if isinstance(_o, type) and hasattr(_o, "_forward_prefill_mha"):
                    cls = _o
                    break
        if cls is not None:
            for pm in ("_forward_prefill_mha", "_forward_prefill_cached_chunked",
                       "_forward_prefill_cached_single_pass"):
                f0 = getattr(cls, pm, None)
                if f0 is None or getattr(f0, "_kvwp", False):
                    continue

                def mkp(orig, name=pm):
                    def wp(*a, **kw):
                        k = "%s@L%d" % (name, _KW["L"])
                        _KVW_PATHS[k] = _KVW_PATHS.get(k, 0) + 1
                        return orig(*a, **kw)
                    wp._kvwp = True
                    return wp

                setattr(cls, pm, mkp(f0))
            print("[kvw] prefill-path counters on %s" % cls.__name__,
                  file=_s2.stderr, flush=True)
    except Exception as _exc:
        print("[kvw] path counters failed: %r" % (_exc,),
              file=sys.stderr, flush=True)
    if n:
        _KW["wrapped"] += n
        print("[kvw] wrapped %d KV-write entry point(s) in "
              "atom.model_ops.attention_mla" % n, file=_s.stderr, flush=True)
    _KW["done"] = True


def _kvw_grab(name, a, kw):
    """Record the KV-write inputs, whole or single-row.

    Signatures differ across the six entry points, so identify by shape: the
    1-D integer tensor is slot_mapping, and the float tensors sharing its dim 0
    are the KV/Q inputs. The paged cache is much larger and never matches.
    """
    vals = list(a) + [kw[k] for k in sorted(kw)]
    names = ["a%02d" % j for j in range(len(a))] + sorted(kw)

    slot, ntok = None, None
    for t in vals:
        if hasattr(t, "dim") and t.dim() == 1 and not t.is_floating_point() \
                and t.numel() >= _KVW_MINTOK:
            slot, ntok = t, t.numel()
            break
    if slot is None:
        return                      # decode / no slot mapping

    row = None
    if _KVW_SLOT >= 0:
        hit = (slot == _KVW_SLOT).nonzero()
        if hit.numel() == 0:
            return                  # this forward does not write our token
        row = int(hit[0, 0])
        # remember the row so the layer hooks can follow the same token
        _KW["row"], _KW["rowfwd"] = row, _KW["fwd"]

    _KW["hits"][name] = _KW["hits"].get(name, 0) + 1
    pre = "f%d_L%02d" % (_KW["fwd"], _KW["L"])
    if row is None:
        _kw_put(pre + "_slot", slot)

    for nm2, t in zip(names, vals):
        if not hasattr(t, "dim") or not hasattr(t, "numel"):
            continue
        if not t.is_floating_point() or t.dim() < 2 or t.shape[0] != ntok:
            continue
        if row is not None:
            _kw_put("%s_%s" % (pre, nm2), t[row])
        elif t.numel() <= 20_000_000:
            _kw_put("%s_%s" % (pre, nm2), t)



def _mqa_bits(t):
    """Raw bytes of a tensor, so fp8 compares bit-exactly."""
    import torch as _t
    v = t.detach().contiguous()
    if v.dtype in (_t.float32, _t.float64, _t.int32, _t.int64):
        return v
    return v.view(_t.uint8)


def _mqa_grab_cu(a, kw, n):
    """Full cu bounds + shapes for one call, keyed by call ordinal."""
    import torch as _t
    g = dict(kw)
    if a:
        for nm, v in zip(("Q", "KV", "kv_scales", "weights",
                          "cu_starts", "cu_ends"), a):
            g.setdefault(nm, v)
    Q, KV = g.get("Q"), g.get("KV")
    cs, ce = g.get("cu_starts"), g.get("cu_ends")
    if Q is None or KV is None or cs is None or ce is None:
        return
    i = _KW.setdefault("cu_n", 0)
    if i >= 800:                     # cover a whole multi-question run
        return
    _KW["cu_n"] = i + 1
    pre = "CU%03d_L%02d" % (i, _KW["L"])
    _kw_put(pre + "_cs", cs[:Q.shape[0]])
    _kw_put(pre + "_ce", ce[:Q.shape[0]])
    _kw_put(pre + "_shape", _t.tensor(
        [Q.shape[0], Q.shape[1], Q.shape[2], KV.shape[0], KV.shape[1]],
        dtype=_t.int64))


def _mqa_grab_inputs(a, kw, n):
    """Record Q/weights/cu bounds for the target row and its own KV window."""
    import torch as _t
    g = dict(kw)
    if a:
        for nm, v in zip(("Q", "KV", "kv_scales", "weights",
                          "cu_starts", "cu_ends"), a):
            g.setdefault(nm, v)
    row = _KVW_LOGITS
    Q, KV = g.get("Q"), g.get("KV")
    ks, w = g.get("kv_scales"), g.get("weights")
    cs, ce = g.get("cu_starts"), g.get("cu_ends")
    if Q is None or KV is None or cs is None or ce is None:
        return
    s, e = int(cs[row]), int(ce[row])
    pre = "MQA_L%02d_n%d" % (_KW["L"], n)
    _kw_put(pre + "_q", _mqa_bits(Q[row]))
    if w is not None:
        _kw_put(pre + "_w", _mqa_bits(w[row]))
    _kw_put(pre + "_kv", _mqa_bits(KV[s:e]))
    if ks is not None:
        _kw_put(pre + "_ks", _mqa_bits(ks[s:e]))
    _kw_put(pre + "_se", _t.tensor([s, e, int(KV.shape[0])], dtype=_t.int64))

def _install_kvw_counter(torch_mod, path):
    import sys as _s
    torch = torch_mod
    _KW["path"] = path

    def hook(module, args, output):
        # guards first: touching tensors during capture/dummy-run is fatal
        try:
            from atom.utils.forward_context import get_forward_context
            if get_forward_context().context.is_dummy_run:
                return
        except Exception:
            pass
        try:
            if torch.cuda.is_current_stream_capturing():
                return
        except Exception:
            pass
        if not _KW["done"]:
            _kvw_try_wrap()
        r = _KW["row"]
        if type(module).__name__ != "DeepseekV2DecoderLayer":
            # sub-module of the layer currently in flight
            if _KVW_SUB >= 0 and r is not None and _KW["rowfwd"] == _KW["fwd"] \
                    and _KW["L"] == _KVW_SUB:
                try:
                    cls = type(module).__name__
                    if cls == "RowParallelLinear" and not _KVW_STACK:
                        _KVW_STACK.append(1)
                        import traceback as _tb
                        import sys as _s3
                        print("[kvw] STACK at layer %d %s:\n%s"
                              % (_KVW_SUB, cls, "".join(_tb.format_stack())),
                              file=_s3.stderr, flush=True)
                    k = "f%d_S%03d_%s" % (_KW["fwd"], _KW["sub"], cls)
                    o = output[0] if isinstance(output, (tuple, list)) else output
                    if hasattr(o, "dim") and o.dim() >= 2 and o.shape[0] > r \
                            and o.numel() // max(o.shape[0], 1) <= 200_000:
                        _kw_put(k + "_out", o[r])
                        _KW["sub"] += 1
                    a0 = args[0] if args else None
                    if hasattr(a0, "dim") and a0.dim() >= 2 and a0.shape[0] > r \
                            and a0.numel() // max(a0.shape[0], 1) <= 200_000:
                        _kw_put(k + "_in", a0[r])
                except Exception:
                    pass
            return
        # the KV write for this layer has already run, so _KW["row"] is the
        # target token's row in this forward
        if r is not None and _KW["rowfwd"] == _KW["fwd"]:
            try:
                pre = "f%d_L%02d_hid" % (_KW["fwd"], _KW["L"])
                o = output[0] if isinstance(output, (tuple, list)) else output
                if hasattr(o, "dim") and o.dim() >= 2 and o.shape[0] > r:
                    _kw_put(pre + "_out", o[r])
                a0 = args[0] if args else None
                if hasattr(a0, "dim") and a0.dim() >= 2 and a0.shape[0] > r:
                    _kw_put(pre + "_in", a0[r])
            except Exception as exc:
                print("[kvw] hid %r" % (exc,), file=_s.stderr, flush=True)
        _KW["L"] += 1
        _KW["sub"] = 0
        if _KW["L"] >= 78:
            _KW["L"] = 0
            _KW["fwd"] += 1
            _KW["row"] = None

    torch_mod.nn.modules.module.register_module_forward_hook(hook)

    import atexit

    def _rep():
        _kw_flush()
        print("[kvw] PATHS %s (pid %d)" % (_KVW_PATHS, os.getpid()),
              file=_s.stderr, flush=True)
        print("[kvw] layer=%d slot=%d wrapped=%d fwds=%d arrays=%d "
              "bytes=%.1fMB hits=%s (pid %d)"
              % (_KVW_L, _KVW_SLOT, _KW["wrapped"], _KW["fwd"],
                 len(_KW["store"]), _KW["bytes"] / 1e6, _KW["hits"], os.getpid()),
              file=_s.stderr, flush=True)
    atexit.register(_rep)


if _KVW:
    _prev_t_kw = _HOOKS.get("torch")

    def _patch_torch_with_kvw(mod, _p=_prev_t_kw):
        if _p is not None:
            _p(mod)
        try:
            _install_kvw_counter(mod, _KVW)
        except Exception as exc:
            print("[kvw] counter failed: %r" % (exc,), file=sys.stderr, flush=True)

    _HOOKS["torch"] = _patch_torch_with_kvw


# --------------------------------------------------------------------------
# GLM_DETTAP=<path> -- per-(forward, layer) bitwise checksum of layer outputs.
# --------------------------------------------------------------------------
_DETTAP = os.environ.get("GLM_DETTAP")
_DETTAP_MAXF = int(os.environ.get("GLM_DETTAP_MAXF", "8192"))
_DETTAP_NL = 80
_DT = {"fwd": 0, "L": 0, "buf": None, "shape": None, "path": None, "over": 0}


def _dt_sum(t):
    """Bitwise int64 checksum: order-independent, so only bit changes show."""
    import torch as _t
    v = t.detach().contiguous()
    if v.dtype in (_t.bfloat16, _t.float16):
        v = v.view(_t.int16)
    elif v.dtype == _t.float32:
        v = v.view(_t.int32)
    elif v.dtype == _t.uint8:
        pass
    else:
        return None
    return v.sum(dtype=_t.int64)


def _install_dettap(torch_mod, path):
    import sys as _s
    torch = torch_mod
    _DT["path"] = path

    def hook(module, args, output):
        try:
            from atom.utils.forward_context import get_forward_context
            if get_forward_context().context.is_dummy_run:
                return
        except Exception:
            pass
        try:
            if torch.cuda.is_current_stream_capturing():
                return
        except Exception:
            pass
        if type(module).__name__ != "DeepseekV2DecoderLayer":
            return
        o = output[0] if isinstance(output, (tuple, list)) else output
        if not hasattr(o, "detach"):
            return
        f, L = _DT["fwd"], _DT["L"]
        if f < _DETTAP_MAXF and L < _DETTAP_NL:
            try:
                if _DT["buf"] is None:
                    _DT["buf"] = torch.zeros(_DETTAP_MAXF, _DETTAP_NL,
                                             dtype=torch.int64, device=o.device)
                    _DT["shape"] = torch.zeros(_DETTAP_MAXF, 2,
                                               dtype=torch.int64, device=o.device)
                c = _dt_sum(o)
                if c is not None:
                    _DT["buf"][f, L] = c
                    if L == 0:
                        _DT["shape"][f, 0] = o.shape[0]
                        _DT["shape"][f, 1] = o.numel()
            except Exception as exc:
                if _DT["over"] < 3:
                    _DT["over"] += 1
                    print("[dettap] %r" % (exc,), file=_s.stderr, flush=True)
        _DT["L"] += 1
        if _DT["L"] >= 78:
            _DT["L"] = 0
            _DT["fwd"] += 1

    torch_mod.nn.modules.module.register_module_forward_hook(hook)
    print("[dettap] installed (max %d forwards)" % _DETTAP_MAXF,
          file=_s.stderr, flush=True)

    import atexit

    def _save():
        import numpy as _np
        if _DT["buf"] is None:
            print("[dettap] nothing captured", file=_s.stderr, flush=True)
            return
        try:
            _np.savez(_DT["path"] + "_%d" % os.getpid(),
                      csum=_DT["buf"].cpu().numpy(),
                      shape=_DT["shape"].cpu().numpy(),
                      nfwd=_np.array([_DT["fwd"]]))
            print("[dettap] saved %d forwards (pid %d)"
                  % (_DT["fwd"], os.getpid()), file=_s.stderr, flush=True)
        except Exception as exc:
            print("[dettap] save failed: %r" % (exc,), file=_s.stderr, flush=True)

    atexit.register(_save)


if _DETTAP:
    _prev_t_dt = _HOOKS.get("torch")

    def _patch_torch_with_dettap(mod, _p=_prev_t_dt):
        if _p is not None:
            _p(mod)
        try:
            _install_dettap(mod, _DETTAP)
        except Exception as exc:
            print("[dettap] hook failed: %r" % (exc,), file=sys.stderr, flush=True)

    _HOOKS["torch"] = _patch_torch_with_dettap


# --------------------------------------------------------------------------
# GLM_DETKERN=1 -- double-execute candidate kernels and compare outputs.
# --------------------------------------------------------------------------
_DETKERN = os.environ.get("GLM_DETKERN") == "1"
_DETKERN_MAX = int(os.environ.get("GLM_DETKERN_MAX", "400"))
_DK = {}


def _dk_rec(name):
    return _DK.setdefault(name, {"calls": 0, "bad": 0, "maxabs": 0.0, "shown": 0})


def _dk_eq(a, b):
    import torch as _t
    if a.dtype.is_floating_point:
        return bool(_t.equal(a.view(_t.int16 if a.element_size() == 2 else _t.int32),
                             b.view(_t.int16 if b.element_size() == 2 else _t.int32)))
    return bool(_t.equal(a, b))


def _dk_wrap_pure(mod, name):
    """For functions that return a fresh tensor and touch nothing in place."""
    import sys as _s
    import torch

    fn = getattr(mod, name, None)
    if fn is None or getattr(fn, "_dk", False):
        return 0

    def w(*a, **kw):
        r1 = fn(*a, **kw)
        rec = _dk_rec(name)
        if rec["calls"] < _DETKERN_MAX and torch.is_tensor(r1):
            rec["calls"] += 1
            try:
                r2 = fn(*a, **kw)
                if not _dk_eq(r1, r2):
                    rec["bad"] += 1
                    d = float((r1.float() - r2.float()).abs().max())
                    rec["maxabs"] = max(rec["maxabs"], d)
                    if rec["shown"] < 3:
                        rec["shown"] += 1
                        print("[detkern] %s NONDETERMINISTIC: max|d|=%.3e shape=%s"
                              % (name, d, tuple(r1.shape)),
                              file=_s.stderr, flush=True)
            except Exception as exc:
                print("[detkern] %s probe failed: %r" % (name, exc),
                      file=_s.stderr, flush=True)
        return r1

    w._dk = True
    setattr(mod, name, w)
    return 1


def _dk_wrap_inplace(mod, name, out_pos):
    """For ops that write their result into positional arg `out_pos`."""
    import sys as _s
    import torch

    fn = getattr(mod, name, None)
    if fn is None or getattr(fn, "_dk", False):
        return 0

    def w(*a, **kw):
        rec = _dk_rec(name)
        probe = (rec["calls"] < _DETKERN_MAX and len(a) > out_pos
                 and torch.is_tensor(a[out_pos]))
        if not probe:
            return fn(*a, **kw)
        o = a[out_pos]
        before = o.detach().clone()
        r = fn(*a, **kw)
        first = o.detach().clone()
        rec["calls"] += 1
        try:
            o.copy_(before)                 # restore, else call 2 sees new state
            fn(*a, **kw)
            if not _dk_eq(first, o.detach()):
                rec["bad"] += 1
                d = float((first.float() - o.float()).abs().max())
                rec["maxabs"] = max(rec["maxabs"], d)
                if rec["shown"] < 3:
                    rec["shown"] += 1
                    print("[detkern] %s NONDETERMINISTIC: max|d|=%.3e shape=%s"
                          % (name, d, tuple(o.shape)), file=_s.stderr, flush=True)
            o.copy_(first)                  # leave the real result in place
        except Exception as exc:
            print("[detkern] %s probe failed: %r" % (name, exc),
                  file=_s.stderr, flush=True)
            o.copy_(first)
        return r

    w._dk = True
    setattr(mod, name, w)
    return 1


def _dk_wrap_module(cls):
    """Double-call a module's forward and compare (forward must be pure)."""
    import sys as _s
    import torch

    name = cls.__name__
    fwd = cls.forward
    if getattr(fwd, "_dk", False):
        return 0

    def w(self, *a, **kw):
        r1 = fwd(self, *a, **kw)
        rec = _dk_rec(name)
        t1 = r1[0] if isinstance(r1, (tuple, list)) and r1 else r1
        if rec["calls"] < _DETKERN_MAX and torch.is_tensor(t1):
            rec["calls"] += 1
            try:
                r2 = fwd(self, *a, **kw)
                t2 = r2[0] if isinstance(r2, (tuple, list)) and r2 else r2
                if torch.is_tensor(t2) and t1.shape == t2.shape and not _dk_eq(t1, t2):
                    rec["bad"] += 1
                    d = float((t1.float() - t2.float()).abs().max())
                    rec["maxabs"] = max(rec["maxabs"], d)
                    if rec["shown"] < 3:
                        rec["shown"] += 1
                        print("[detkern] %s NONDETERMINISTIC: max|d|=%.3e shape=%s"
                              % (name, d, tuple(t1.shape)),
                              file=_s.stderr, flush=True)
            except Exception as exc:
                print("[detkern] %s probe failed: %r" % (name, exc),
                      file=_s.stderr, flush=True)
        return r1

    w._dk = True
    cls.forward = w
    return 1


def _dk_wrap_decode(mod, name, out_pos, scratch):
    """In-place op with extra scratch buffers to snapshot/restore."""
    import sys as _s
    import torch

    fn = getattr(mod, name, None)
    if fn is None or getattr(fn, "_dk", False):
        return 0

    def w(*a, **kw):
        rec = _dk_rec(name)
        pos = [out_pos] + [p for p in scratch if len(a) > p and torch.is_tensor(a[p])]
        probe = (rec["calls"] < _DETKERN_MAX and len(a) > out_pos
                 and torch.is_tensor(a[out_pos]))
        if not probe:
            return fn(*a, **kw)
        saved = {p: a[p].detach().clone() for p in pos}
        r = fn(*a, **kw)
        first = a[out_pos].detach().clone()
        rec["calls"] += 1
        try:
            for p in pos:
                a[p].copy_(saved[p])
            fn(*a, **kw)
            if not _dk_eq(first, a[out_pos].detach()):
                rec["bad"] += 1
                d = float((first.float() - a[out_pos].float()).abs().max())
                rec["maxabs"] = max(rec["maxabs"], d)
                if rec["shown"] < 3:
                    rec["shown"] += 1
                    print("[detkern] %s NONDETERMINISTIC: max|d|=%.3e shape=%s"
                          % (name, d, tuple(first.shape)),
                          file=_s.stderr, flush=True)
        except Exception as exc:
            print("[detkern] %s probe failed: %r" % (name, exc),
                  file=_s.stderr, flush=True)
        a[out_pos].copy_(first)
        return r

    w._dk = True
    setattr(mod, name, w)
    return 1


def _install_detkern():
    import sys as _s
    n = 0
    am = _s.modules.get("atom.model_ops.attention_mla")
    ds = _s.modules.get("atom.models.deepseek_v2")
    if am is not None:
        n += _dk_wrap_inplace(am, "mla_prefill_fwd", 2)
        n += _dk_wrap_inplace(am, "mla_decode_fwd", 2)
        n += _dk_wrap_pure(am, "gather_kv_b_proj")
    if ds is not None:
        n += _dk_wrap_pure(ds, "fp8_mqa_logits")
    if am is not None:
        n += _dk_wrap_inplace(am, "triton_shuffle_mla_decode_fwd", 2)
        try:
            import importlib
            _md = importlib.import_module("aiter.ops.triton.attention.mla_decode")
            n += _dk_wrap_decode(_md, "decode_attention_fwd", 3, (4, 7))
        except Exception as _e:
            print("[detkern] decode wrap: %r" % (_e,), file=_s.stderr, flush=True)
    # MoE and attention at module level (their forwards return fresh tensors)
    seen = set()
    for _m in list(_s.modules.values()):
        if _m is None:
            continue
        for _cn in ("DeepseekV2MoE", "FusedMoE"):
            _c = getattr(_m, _cn, None)
            if isinstance(_c, type) and _cn not in seen and hasattr(_c, "forward"):
                seen.add(_cn)
                n += _dk_wrap_module(_c)
    print("[detkern] wrapped %d op(s)" % n, file=_s.stderr, flush=True)
    return n


if _DETKERN:
    for _mn in ("atom.model_ops.attention_mla", "atom.models.deepseek_v2"):
        _prev_dk = _HOOKS.get(_mn)

        def _mk(_p=_prev_dk):
            def _h(mod):
                if _p is not None:
                    _p(mod)
                try:
                    _install_detkern()
                except Exception as exc:
                    print("[detkern] hook failed: %r" % (exc,), file=sys.stderr,
                          flush=True)
            return _h

        _HOOKS[_mn] = _mk()

    import atexit as _ax_dk
    _ax_dk.register(lambda: print(
        "[detkern] SUMMARY " + " | ".join(
            "%s: %d calls, %d nondeterministic (max|d|=%.3e)"
            % (k, v["calls"], v["bad"], v["maxabs"]) for k, v in sorted(_DK.items()))
        + " (pid %d)" % os.getpid(), file=sys.stderr, flush=True))


# --------------------------------------------------------------------------
# GLM_TRITON_A16W16=1 -- bf16 GEMM via aiter Triton instead of Tensile.
# --------------------------------------------------------------------------
# Default ON: 3-6x on the bf16 attention projections, which are 41% of
# device time. Full-set gsm8k accuracy is identical to the Tensile path.
# Set GLM_TRITON_A16W16=0 to fall back.
_A16W16 = os.environ.get("GLM_TRITON_A16W16", "1") == "1"
_A16W16_MINFLOP = float(os.environ.get("GLM_A16W16_MINFLOP", "5e9"))
_A16W16_MAXOUT = 2 ** 31            # int32 offset overflow above this
_AW = {"tri": 0, "torch": 0, "err": 0}


def _install_a16w16(aiter_mod):
    import sys as _s
    import torch

    try:
        from aiter.ops.triton.gemm.basic.gemm_a16w16 import (
            gemm_a16w16 as _tri_gemm,
        )
    except Exception as exc:
        print("[a16w16] triton kernel unavailable: %r" % (exc,),
              file=_s.stderr, flush=True)
        return

    import aiter.tuned_gemm as tg
    orig = getattr(tg, "gemm_a16w16", None)
    if orig is None or getattr(orig, "_a16w16", False):
        return

    def w(A, B, bias=None, otype=None, scale_a=None, scale_b=None,
          scale_c=None):
        try:
            if (scale_a is None and scale_b is None and scale_c is None
                    and A.dtype in (torch.bfloat16, torch.float16)
                    and B.dtype == A.dtype and A.dim() == 2
                    and (otype is None or otype == A.dtype)):
                M, K = A.shape
                N = B.shape[0]
                if M * N < _A16W16_MAXOUT and 2.0 * M * N * K >= _A16W16_MINFLOP:
                    _AW["tri"] += 1
                    return _tri_gemm(A, B, bias=bias, dtype=A.dtype)
        except Exception as exc:
            _AW["err"] += 1
            if _AW["err"] <= 3:
                print("[a16w16] fell back: %r" % (exc,), file=_s.stderr,
                      flush=True)
        _AW["torch"] += 1
        return orig(A, B, bias, otype, scale_a, scale_b, scale_c)

    w._a16w16 = True
    tg.gemm_a16w16 = w
    # TunedGemm.mm captured the module-level name at def time in some builds;
    # rebind anywhere it was imported by value.
    for mn, m in list(_s.modules.items()):
        if m is None or not (mn == "aiter" or mn.startswith("aiter.")):
            continue
        if getattr(m, "gemm_a16w16", None) is orig:
            setattr(m, "gemm_a16w16", w)
    print("[a16w16] triton bf16 GEMM enabled (minflop=%.1e)" % _A16W16_MINFLOP,
          file=_s.stderr, flush=True)


if _A16W16:
    _prev_a_aw = _HOOKS.get("aiter")

    def _aiter_with_a16w16(mod, _p=_prev_a_aw):
        if _p is not None:
            _p(mod)
        try:
            _install_a16w16(mod)
        except Exception as exc:
            print("[a16w16] hook failed: %r" % (exc,), file=sys.stderr,
                  flush=True)

    _HOOKS["aiter"] = _aiter_with_a16w16

    import atexit as _ax_aw
    _ax_aw.register(lambda: print(
        "[a16w16] SUMMARY triton=%d torch=%d fallback_errors=%d (pid %d)"
        % (_AW["tri"], _AW["torch"], _AW["err"], os.getpid()),
        file=sys.stderr, flush=True))


# --------------------------------------------------------------------------
# GLM_WHICH_IMPL=1 -- one-shot report of the live MoE GEMM and its scale layout.
#
# Which kernel the MoE lands on is decided by config (act_quant), and the scale
# layout by arch, so static reading gives two answers that have to agree. This
# prints the pair on the first call. On GLM-5.2 it reports
#   GEMM=moe_gemm_a4w4 swizzle_mx_scale='CDNA4_SCALE' x.dtype=torch.uint8
# i.e. the gfx950 layout on gfx1250 hardware, which is what the scale_arch pin
# in fused_moe_triton.py exists to produce.
# --------------------------------------------------------------------------
_WHICHIMPL = os.environ.get("GLM_WHICH_IMPL") == "1"
_WI = {"done": set()}


def _install_whichimpl(mod):
    import sys as _s
    ds = _s.modules.get("atom.model_ops.fused_moe_triton")
    if ds is None:
        return
    for nm in ("moe_gemm_a4w4", "moe_gemm_a16w4"):
        fn = getattr(ds, nm, None)
        if fn is None or getattr(fn, "_wi", False):
            continue

        def mk(orig, name=nm):
            def w(*a, **kw):
                if name not in _WI["done"]:
                    _WI["done"].add(name)
                    sw = kw.get("swizzle_mx_scale")
                    x = a[0] if a else None
                    print("[whichimpl] GEMM=%s swizzle_mx_scale=%r x.dtype=%s "
                          "out_dtype=%r" % (name, sw,
                                            getattr(x, "dtype", "?"),
                                            kw.get("out_dtype")),
                          file=_s.stderr, flush=True)
                return orig(*a, **kw)
            w._wi = True
            return w
        setattr(ds, nm, mk(fn))
    sh = _s.modules.get("aiter.ops.triton.utils.shuffle")
    if sh is not None and not getattr(sh.shuffle_scale_moe, "_wi", False):
        o = sh.shuffle_scale_moe

        def w2(*a, **kw):
            r = o(*a, **kw)
            if "shuffle" not in _WI["done"]:
                _WI["done"].add("shuffle")
                lab = r[1] if isinstance(r, tuple) and len(r) > 1 else None
                print("[whichimpl] shuffle_scale_moe(arch=%r) -> layout=%r"
                      % (kw.get("arch"), lab), file=_s.stderr, flush=True)
            return r
        w2._wi = True
        sh.shuffle_scale_moe = w2
    print("[whichimpl] installed", file=_s.stderr, flush=True)


if _WHICHIMPL:
    _prev_wi = _HOOKS.get("atom.model_ops.fused_moe_triton")

    def _wi_hook(mod, _p=_prev_wi):
        if _p is not None:
            _p(mod)
        try:
            _install_whichimpl(mod)
        except Exception as exc:
            print("[whichimpl] failed: %r" % (exc,), file=sys.stderr, flush=True)

    _HOOKS["atom.model_ops.fused_moe_triton"] = _wi_hook
