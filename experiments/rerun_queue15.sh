#!/usr/bin/env bash
# Fifteenth CPU queue (14 Oct): P2-N on Qwen3.5-2B, then Qwen3.5-0.8B
# (docs/preregistration-p2n-qwen35-current.md), then P2-H-1B on Llama-3.2-1B-Instruct
# (docs/preregistration-p2h1b-llama1b-cpu.md), which waits for gated access to be granted.
# Every run resumes by key; a complete file is skipped.
case "$0" in
  */*) cd "${0%/*}/.." ;;
esac
export PYTHONPATH=.
LOG=results/rerun_queue15.log
say () { printf '[%(%Y-%m-%d %H:%M)T] %s\n' -1 "$*" >> "$LOG"; }
snap () { python -c "from huggingface_hub import snapshot_download as s; print(s('$1'))" 2>/dev/null | tail -1; }
mkdir -p results/p2n results/p2h1b

p2n () {  # p2n <tag> <hub id>
  local tag=$1 path
  path=$(snap "$2")
  [ -d "$path" ] || { say "P2-N $tag: download of $2 failed"; return; }
  local base=(--model qwen3b --model-path "$path" --quant none --compute-dtype fp32 --alpha-frac 0 0.25 0.5 1.0)
  for spec in "concept|--vector-pos concept" "tail|--vector-pos template-tail" "sentence|--vector-recipe aperture"; do
    local arm=${spec%%|*} flag=${spec#*|}
    for stage in steer forced; do
      say "starting P2-N $tag $arm $stage"
      python experiments/kaggle_s3_positive_control.py "${base[@]}" --stage $stage $flag \
        --out results/p2n/s2_${tag}_${arm}.jsonl >> results/p2n/${tag}.log 2>&1
      say "P2-N $tag $arm $stage exit $?"
    done
  done
  say "starting P2-N $tag random forced"
  python experiments/kaggle_s3_positive_control.py "${base[@]}" --stage forced --control random \
    --out results/p2n/s2_${tag}_random.jsonl >> results/p2n/${tag}.log 2>&1
  say "P2-N $tag random forced exit $?"
}

p2n qwen35_2b Qwen/Qwen3.5-2B
p2n qwen35_08b Qwen/Qwen3.5-0.8B
say "P2-N done"

until python -c "from huggingface_hub import hf_hub_download as h; h('meta-llama/Llama-3.2-1B-Instruct', 'config.json')" >/dev/null 2>&1; do
  say "P2-H-1B: no access to meta-llama/Llama-3.2-1B-Instruct yet; retry in 30 min"
  sleep 1800
done
path=$(snap meta-llama/Llama-3.2-1B-Instruct)
say "starting P2-H-1B ($path)"
python experiments/kaggle_p2h_hahami.py --model-path "$path" --quant none --compute-dtype fp32 --layers 0 10 \
  --alphas 5 3 1 2 4 8 12 --no-fidelity --out results/p2h1b/p2h1b_llama1b.jsonl >> results/p2h1b/p2h1b.log 2>&1
say "P2-H-1B exit $?"
say "queue 15 done"
