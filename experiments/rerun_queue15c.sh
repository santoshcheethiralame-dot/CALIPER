#!/usr/bin/env bash
# Queue 15, last part (15 Oct). Queue 15b finished Qwen3.5-2B; its 0.8B and P2-H-1B runs failed in
# a minute because both downloads failed that evening and snap() fell back to a cached folder that
# held config.json only. Here a snapshot counts only if it holds weights. Same commands otherwise.
case "$0" in
  */*) cd "${0%/*}/.." ;;
esac
export PYTHONPATH=. HF_HUB_DISABLE_XET=1
LOG=results/rerun_queue15.log
say () { printf '[%(%Y-%m-%d %H:%M)T] %s\n' -1 "$*" >> "$LOG"; }
has_weights () { ls "$1"/*.safetensors >/dev/null 2>&1; }
snap () {  # up to 6 tries; the local copy only if it holds weights
  local p
  for i in 1 2 3 4 5 6; do
    p=$(python -c "from huggingface_hub import snapshot_download as s; print(s('$1', ignore_patterns=['original/*']))" 2>/dev/null | tail -1)
    [ -d "$p" ] && has_weights "$p" && { echo "$p"; return; }
    sleep 120
  done
  p=$(python -c "from huggingface_hub import snapshot_download as s; print(s('$1', local_files_only=True))" 2>/dev/null | tail -1)
  [ -d "$p" ] && has_weights "$p" && echo "$p"
}
rows () { [ -f "$1" ] && grep -c . "$1" || echo 0; }
say "queue 15c starts (Qwen3.5-0.8B, then P2-H-1B; a snapshot must hold weights)"

tag=qwen35_08b
path=$(snap Qwen/Qwen3.5-0.8B)
if [ -z "$path" ]; then
  say "P2-N $tag: no weights for Qwen/Qwen3.5-0.8B"
else
  base=(--model qwen3b --model-path "$path" --quant none --compute-dtype fp32 --alpha-frac 0 0.25 0.5 1.0)
  for spec in "concept|--vector-pos concept|concept" "tail|--vector-pos template-tail|tail" \
              "sentence|--vector-recipe aperture|sentence_STAGE_aperture" "random|--control random|random_STAGE_random"; do
    IFS='|' read -r arm flag stem <<< "$spec"
    for stage in steer forced; do
      [ "$arm" = random ] && [ $stage = steer ] && continue
      f=results/p2n/s2_${tag}_${stem//_STAGE/_$stage}
      [ "$stem" = "$arm" ] && f=results/p2n/s2_${tag}_${arm}_${stage}
      want=120; extra=()
      [ $stage = forced ] && want=240 extra=(--no-health)
      if [ "$(rows ${f}_none.jsonl)" -ge $want ]; then say "P2-N $tag $arm $stage complete, skipped"; continue; fi
      say "starting P2-N $tag $arm $stage"
      python experiments/cpu_fp32_lowmem.py "${base[@]}" --stage $stage $flag "${extra[@]}" \
        --out results/p2n/s2_${tag}_${arm}.jsonl >> results/p2n/${tag}.log 2>&1
      say "P2-N $tag $arm $stage exit $?"
    done
  done
fi
say "P2-N done"

path=$(snap meta-llama/Llama-3.2-1B-Instruct)
if [ -n "$path" ]; then
  say "starting P2-H-1B ($path)"
  python experiments/kaggle_p2h_hahami.py --model-path "$path" --quant none --compute-dtype fp32 --layers 0 10 \
    --alphas 5 3 1 2 4 8 12 --no-fidelity --out results/p2h1b/p2h1b_llama1b.jsonl >> results/p2h1b/p2h1b.log 2>&1
  say "P2-H-1B exit $?"
else
  say "P2-H-1B: no weights for meta-llama/Llama-3.2-1B-Instruct"
fi
say "queue 15 done"
