# gfx1250 bringup shim

Scaffolding that lets GLM-5.2 (MXFP4) run on gfx1250, where a large part of
aiter's CK/HIP kernel set does not build. Committed as-is from the bringup
session so it stops living only in a container; **it is not production code and
is expected to be cleaned up**, see "Cleanup" below.

## How it activates

`sitecustomize.py` is auto-imported by CPython when its directory is on
`PYTHONPATH`:

    export PYTHONPATH=/path/to/tools/gfx1250-shim${PYTHONPATH:+:$PYTHONPATH}
    export GFX1250_SHIM=1
    export GFX1250_SHIM_LOG=/path/to/missing_kernels.jsonl   # optional

It installs a meta-path finder keyed on module name (`_HOOKS`), so each hook
fires the first time its module is imported. Hooks compose by calling the
previous one. A hook that needs aiter must be keyed on `"aiter"`, not `"torch"` --
aiter is not imported yet when the torch hook runs.

## Why it exists

`module_cache` and friends fail to compile on gfx1250: CK-tile's supported-arch
list has no gfx1250 entry, so the build dies with

    ck_tile/core/config.hpp:577: "Only one target architecture can be defined"
    error: use of undeclared identifier 'CK_TILE_BUFFER_RESOURCE_3RD_DWORD'

The shim intercepts `compile_ops` and supplies replacements.

## What it replaces

Genuine torch/python fallbacks -- no native kernel exists:

| op | note |
| --- | --- |
| `top_k_per_row_prefill` / `_decode` | DSA indexer top-k. Hot: ~1.7% of device time. |
| `concat_and_cache_mla` | MLA KV scatter. Only reached on the dense path (seq <= index_topk), which is why short prompts used to abort and long ones did not. |
| `cp_gather_indexer_k_quant_cache` | gathers indexer KV out of the paged cache |
| `get_mla_metadata_v1` | host-side scheduling metadata |
| `greedy_sample`, `mixed_sample_outer_exponential` | sampling |

Marshalling adapters over real Triton/Gluon kernels -- these are *not* torch
reimplementations:

| op | routes to |
| --- | --- |
| `mla_prefill_asm_fwd` | `unified_attention_sparse_mla`; converts CSR top-k to dense `[T,K]` and re-views page_size=1 KV as 64-slot tiles |
| `flash_attn_varlen_func` | Triton `mha`; drops the `min_seqlen_q` CK scheduler hint |
| `indexer_k_quant_and_cache`, `rope_cached_positions_2c_fwd_inplace`, `dynamic_per_token_scaled_quant`, `silu_and_mul` | aiter Triton equivalents |

Plus ~38 module-level redirects (rmsnorm, layernorm, rope, gemm_a8w8, batched
GEMMs, mxfp4 quant, causal_conv1d) from the CK/HIP default to aiter Triton.

## Production flags

| env | default | effect |
| --- | --- | --- |
| `GLM_TRITON_SPARSE_MLA` | off | sparse-MLA prefill via `unified_attention_sparse_mla` instead of the torch reference. Should be on. |
| `GLM_ORDERED_TOPK` | **1** | DSA top-k ordered by (score desc, token asc). `torch.topk` breaks ties arbitrarily and its choice moves with tensor shape, which made the selected order depend on batch composition. |
| `GLM_TRITON_A16W16` | off | bf16 GEMMs via the gfx1250 Gluon `gemm_a16w16` instead of falling through `tuned_gemm` to `F.linear` -> hipBLASLt Tensile. Measured on the full gsm8k set: 15-shot TTFT -12.2%, TPOT -8.8%; accuracy unchanged (1233/1319 both ways). Left off by default pending the determinism note below. |
| `GLM_A16W16_MINFLOP` | 5e9 | work-size gate for the above. Not arbitrary: the Gluon path has a ~76 us fixed floor and Tensile runs ~70 TF/s, so they cross at ~5.3e9 FLOPs. |
| `GLM_MQA_TAIL` | off | **obsolete.** Repaired the window tail that gfx1250's gluon `fp8_mqa_logits` skipped; fixed properly in aiter (`kv_pos_post`, commit dd408f6f2), after which it repairs 0 columns. |
| `GLM_FQK_TRITON` | off | fused qk-rope/cache via Triton |

## Diagnostic flags -- delete these in cleanup

Everything below is investigation scaffolding, all env-gated and inert by
default: `GLM_KVW*`, `GLM_DETTAP`, `GLM_DETKERN`, `GLM_IDX_CHECK`, `GLM_MQA_IN`,
`GLM_MQA_CU`, `GLM_MQA_COUNT`, `GLM_ATTNTAP`, `GLM_LAYERTAP`, `GLM_TSM_DIFF`,
`GLM_PCTAP`, `GLM_MOETAP`, `GLM_KVWATCH`, `GLM_NO_GLUON_MQA`.

They cost nothing when unset but are the bulk of the file's size.

## Known issues

* **`ATOM_USE_TRITON_MLA_SHUFFLE_KV=1` is unsafe at long context.** The sparse
  attention reader mishandles the shuffled KV layout: 15-shot gsm8k returns pure
  token garbage (0/12), deterministically, while 5-shot (dense path) is fine.
  `env_triton.sh` defaults this to 1. Use `shuffle_kv=0`.
* **`GLM_TRITON_A16W16` costs determinism.** With it on, an A/A pair (identical
  config, repeated) diverged on 4 of 60 questions where the baseline diverges on
  0-1. Full-set accuracy is unaffected.
* **Rare run-level nondeterminism at `max_num_seqs >= 2`**, independent of this
  shim and of the aiter fixes: ~3 of 8 repeats perturb one question. seqs=1 is
  bit-reproducible.
* The `gemm_a16w16` Triton kernel corrupts its output above 2^31 elements
  (err 6.9e-01 vs 9.8e-04). Guarded by `M*N < 2**31`; not reachable at this
  model's shapes. Root cause not established -- it is *not* the output offsets
  (already int64) and *not* split-K (`NUM_KSPLIT=1` at the failing shape).

## Harness

`harness/` holds the launcher the results in this README were produced with:

* `go_g8.sh` -- arm runner: `$1 gpu $2 fewshot $3 shuffle_kv $4 triton_sparse_mla
  $5 prefix_caching $6 limit $7 logname`. Also suppresses GPU coredumps, which
  matters: a fault with the model resident dumps ~445 GB and can fill the disk.
* `env_triton.sh` -- the `ATOM_USE_TRITON_*` selection. **Note
  `ATOM_USE_TRITON_MLA_SHUFFLE_KV` is set to 0 here**; upstream defaults it to 1,
  which is unsafe at long context (see Known issues).
* `run_gsm8k.py` -- gsm8k harness with `--only <file>` (run specific test indices,
  dumped under their original index) and `--chunk N` (checkpoint every N).

## Stack the results were measured on

| component | version |
| --- | --- |
| aiter | branch `glm52-gfx1250-fixes` @ `6058cca7b` (NaN guard + `kv_pos_post`) |
| ATOM | branch `lijinpei/glm52-gfx1250` |
| triton | `3.8.0+gitb88cf29a` -- **pinned**. An earlier build (`5b5a3760`) made runs non-reproducible: 1/18 identical vs 18/18 on b88cf29a. |
| torch | `2.11.0+rocm7.15.0a20260712` |
| model | GLM-5.2-MXFP4 (78 layers, `index_topk` 2048, `index_topk_freq` 4, 256 experts, 8 active) |

`3rdparty/composable_kernel` carries an uncommitted submodule pointer change on
both bringup boxes; it has not been established whether that is load-bearing.

## Cleanup path

1. Drop the diagnostic block wholesale.
2. Promote the genuine fallbacks (`top_k_per_row_*`, `concat_and_cache_mla`,
   `cp_gather_indexer_k_quant_cache`) into ATOM proper as arch-gated ops, or get
   native kernels for them -- the top-k one is measurably hot.
3. Fold the marshalling adapters into the aiter call sites they wrap.
4. Resolve the `GLM_TRITON_A16W16` determinism regression, then flip the default.
