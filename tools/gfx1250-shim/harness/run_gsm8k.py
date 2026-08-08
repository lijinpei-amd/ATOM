import os
"""GSM8K few-shot accuracy for GLM-5.2 on gfx1250.

Offline: reads the parquet straight out of the HF hub cache. Standard
lm-eval-style prompt ("Question: ... Answer: ..." blocks, greedy, stop at the
next "Question:"), answer = the number after "####" if the model emits one,
else the last number in the completion.
"""
import argparse
import glob
import re
import time

from atom import SamplingParams
from atom.model_engine.arg_utils import EngineArgs
try:
    from atom.utils.arg_parser import FlexibleArgumentParser
except ModuleNotFoundError:          # older ATOM: same flags, plain parser
    from argparse import ArgumentParser as FlexibleArgumentParser

GSM = os.environ.get(
    "GSM8K_GLOB",
    "/data/huggingface-cache/hub/datasets--openai--gsm8k/snapshots/"
    "*/main/%s-00000-of-00001.parquet",
)
_ORIG_IDX = None
NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def load(split):
    import pyarrow.parquet as pq
    t = pq.read_table(sorted(glob.glob(GSM % split))[0]).to_pydict()
    return list(zip(t["question"], t["answer"]))


def norm(s):
    s = s.replace(",", "").rstrip(".")
    try:
        f = float(s)
    except ValueError:
        return None
    return int(f) if f == int(f) else f


def extract(text):
    if "####" in text:
        m = NUM.search(text.split("####", 1)[1])
        if m:
            return norm(m.group())
    m = NUM.findall(text)
    return norm(m[-1]) if m else None



def _score(out, gold):
    """(n_correct, records) for however many outputs we have so far."""
    ok = 0
    recs = []
    for i, (o, g) in enumerate(zip(out, gold)):
        txt = o["text"] if isinstance(o, dict) else str(o)
        pred = extract(txt)
        hit = pred is not None and g is not None and pred == g
        ok += hit
        gi = _ORIG_IDX[i] if _ORIG_IDX is not None else i
        recs.append({"i": gi, "gold": g, "pred": pred, "correct": bool(hit),
                     "ntok": len(txt), "text": txt[-300:]})
    return ok, recs


def _score_and_dump(args, out, gold, partial=False):
    if not args.dump:
        return
    import json
    ok, recs = _score(out, gold)
    tmp = args.dump + ".tmp"
    with open(tmp, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    os.replace(tmp, args.dump)          # atomic: never a half-written dump
    if partial:
        print(f"[ckpt] {ok}/{len(recs)} correct so far "
              f"({100.0 * ok / max(len(recs), 1):.2f}%)", flush=True)


def main():
    p = FlexibleArgumentParser(description="GLM-5.2 gsm8k")
    EngineArgs.add_cli_args(p)
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--only", type=str, default=None,
                   help="file of test indices (one per line); run only those, "
                        "and dump them under their ORIGINAL index")
    p.add_argument("--num-fewshot", type=int, default=5)
    p.add_argument("--gen-tokens", type=int, default=256)
    p.add_argument("--show", type=int, default=3)
    p.add_argument("--chunk", type=int, default=0,
                   help="generate in batches of N and checkpoint --dump after "
                        "each; 0 = one generate() call over everything")
    p.add_argument("--dump", type=str, default=None,
                   help="write per-question results as jsonl (for paired tests)")
    args = p.parse_args()

    train, test = load("train"), load("test")
    shots = "".join(f"Question: {q}\nAnswer: {a}\n\n"
                    for q, a in train[: args.num_fewshot])
    global _ORIG_IDX
    if args.only:
        keep = [int(x) for x in open(args.only).read().split()]
        items = [test[k] for k in keep]
        _ORIG_IDX = keep
        print(f"--only: {len(keep)} selected indices", flush=True)
    else:
        items = test[: args.limit]
        _ORIG_IDX = list(range(len(items)))
    prompts = [shots + f"Question: {q}\nAnswer:" for q, _ in items]
    gold = [norm(a.split("####")[-1].strip()) for _, a in items]
    print(f"gsm8k: {len(items)} questions, {args.num_fewshot}-shot, "
          f"prompt chars ~{len(prompts[0])}", flush=True)

    llm = EngineArgs.from_cli_args(args).create_engine()
    sp = SamplingParams(
        temperature=0.0, max_tokens=args.gen_tokens,
        stop_strings=["\nQuestion:", "\n\nQuestion:"])
    t0 = time.time()
    if args.chunk and args.chunk > 0:
        # Same prompts, same order, same sampling -- only the queue is handed
        # over in slices so partial results can be flushed. max_num_seqs still
        # caps real concurrency, so this changes drain pattern, not semantics.
        out = []
        for _s in range(0, len(prompts), args.chunk):
            out.extend(llm.generate(prompts[_s:_s + args.chunk], sp))
            _score_and_dump(args, out, gold, partial=True)
            print(f"[ckpt] {len(out)}/{len(prompts)} done, "
                  f"{time.time() - t0:.0f}s", flush=True)
    else:
        out = llm.generate(prompts, sp)
    dt = time.time() - t0

    ok = 0
    recs = []
    for i, (o, g) in enumerate(zip(out, gold)):
        txt = o["text"] if isinstance(o, dict) else str(o)
        pred = extract(txt)
        hit = pred is not None and g is not None and pred == g
        ok += hit
        gi = _ORIG_IDX[i] if _ORIG_IDX is not None else i
        recs.append({"i": gi, "gold": g, "pred": pred, "correct": bool(hit),
                     "ntok": len(txt), "text": txt[-300:]})
        if i < args.show:
            print(f"\n--- q{i} gold={g} pred={pred} {'OK' if hit else 'MISS'}\n"
                  f"{txt.strip()[:400]}", flush=True)
    if args.dump:
        import json
        with open(args.dump, "w") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
        print(f"per-question results -> {args.dump}", flush=True)
    print(f"\n{'='*60}\nGSM8K {args.num_fewshot}-shot accuracy: "
          f"{ok}/{len(items)} = {100.0*ok/len(items):.2f}%   ({dt:.1f}s)\n{'='*60}",
          flush=True)


if __name__ == "__main__":
    main()
