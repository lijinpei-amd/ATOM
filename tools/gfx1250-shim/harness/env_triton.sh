# Max-triton/gluon kernel selection for gfx1250 discovery
export ATOM_USE_TRITON_GEMM=1
export ATOM_USE_TRITON_MXFP4_BMM=1
export ATOM_USE_TRITON_MLA=1
# Shuffled KV is UNSAFE at long context: the sparse-MLA attention reader
# mishandles the layout and returns token garbage (15-shot gsm8k 0/12,
# deterministic); the dense path (seq <= index_topk) is unaffected.
export ATOM_USE_TRITON_MLA_SHUFFLE_KV=0
export ATOM_USE_TRITON_MOE=1
# gfx1250 gluon a8w4 GGUU decode MoE: process_weights_after_loading returns early
# on the use_triton (a4w4) branch, so w13_weight_preshuffled is never created,
# yet apply() still takes the decode branch -> AttributeError. Mutually exclusive.
export ATOM_USE_TRITON_MOE_DECODE=0
export ATOM_FORCE_ATTN_TRITON=1
export ATOM_USE_FP4_NON_SHUFFLE_TRITON_GEMM=1
export HF_HUB_OFFLINE=1
export ATOM_LOG_MORE=1

# gfx1250 kernel-discovery shim (prefer triton, log missing CK/HIP kernels)
export GFX1250_SHIM=1
export GFX1250_SHIM_LOG=/home/jinpli/workspace/glm52/logs/missing_kernels.jsonl
export PYTHONPATH=/home/jinpli/workspace/glm52/shim${PYTHONPATH:+:$PYTHONPATH}

# Disable ATOM's CK/HIP fused kernels (no gfx1250 build) to fall back to the
# unfused triton/torch paths. Each of these is a "missing kernel" in its own right.
export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=0
export ATOM_ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION=0
export ATOM_ENABLE_DS_QKNORM_QUANT_FUSION=0
export ATOM_ENABLE_DS_QKNORM_FUSION=0
export ATOM_ENABLE_DS_INDEXER_QK_ROPE_CACHE_FUSION=0
export ATOM_ENABLE_GLM_FUSED_INDEXER=0
export ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION=0
