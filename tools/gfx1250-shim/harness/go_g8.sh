#!/bin/bash
# Generalized gsm8k arm runner.
#  $1 GPU  $2 fewshot  $3 shuffle_kv  $4 triton_sparse_mla  $5 prefix_caching(on|off)
#  $6 limit  $7 logname
cd /home/jinpli/workspace/glm52
source ./env_triton.sh
# A GPU fault with the model resident dumps ~445 GB (all of VRAM) and can fill
# the disk. Belt and braces: rlimit, the HSA switch, and a /dev/null pattern.
ulimit -c 0 2>/dev/null || true
export HSA_ENABLE_COREDUMP=0
export HSA_COREDUMP_PATTERN=/dev/null
export AMD_COREDUMP=0
export ATOM_LOADER_NUM_THREADS=1
export HIP_VISIBLE_DEVICES=$1
FEWSHOT=$2
export ATOM_USE_TRITON_MLA_SHUFFLE_KV=$3
export GLM_FQK_TRITON=1
export GLM_DECBUF_FILL=${DECFILL:-0}
export GLM_ZERO_DECBUF=${ZDEC:-0}
export GLM_KVWATCH=${KVWATCH:-0}
export GLM_KVSCAN=${KVSCAN:-0}
export GLM_ZERO_KV=${ZEROKV:-0}
export GLM_BLOWUP=${BLOWUP:-0}
export GLM_KVGUARD=${KVGUARD:-0}
export GLM_SAMPLE_CHECK=${SMPCHK:-0}
export GLM_ATTN_NOISE=${ATTNNOISE:-0}
export GLM_BMM_OWNY=${OWNY:-0}
export GLM_BMM_CONTIG=${BMMCONTIG:-0}
export GLM_MQA_TAIL=${MQATAIL:-0}
export GLM_NO_GLUON_MQA=${NOGLU:-0}
if [ "$4" = "1" ]; then export GLM_TRITON_SPARSE_MLA=1; fi
PC=""
if [ "$5" = "off" ]; then PC="--no-enable_prefix_caching"; fi
if [ "${8:-0}" = "1" ]; then export GLM_SPARSE_DECODE=1; fi
LIMIT=$6; LOG=$7
echo "ARM gpu=$1 fewshot=$FEWSHOT shufkv=$3 tritonsparse=$4 pc=$5 limit=$LIMIT" > logs/$LOG
timeout 14400 python3 run_gsm8k.py \
  --model /home/jinpli/workspace/models/GLM-5.2-MXFP4 \
  ${EAGER:+--enforce-eager} ${ONLY:+--only $ONLY} --chunk ${CHUNK:-0} --limit $LIMIT --num-fewshot $FEWSHOT --gen-tokens 256 --dump $PWD/${LOG%.log}.jsonl \
  -tp 1 --level 0 --cudagraph-mode NONE $PC \
  --block-size 64 --max-model-len 4096 --max-num-batched-tokens ${MAXBT:-4096} \
  --max-num-seqs ${SEQS:-6} --gpu-memory-utilization 0.985 >> logs/$LOG 2>&1
echo "EXIT=$?" >> logs/$LOG
