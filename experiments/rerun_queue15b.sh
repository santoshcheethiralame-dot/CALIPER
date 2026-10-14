#!/usr/bin/env bash
# Queue 15, continued (14 Oct). Queue 15 rebuilt the vectors and recomputed every health check
# in each forced run; analyse_s2.py reads health only from the steer configs, so the forced runs
# here pass --no-health (the forced rows are unchanged). A stage whose file already holds all
# its rows is skipped before the model loads. Takes over once queue 15's first run ends.
case "$0" in
  */*) cd "${0%/*}/.." ;;
esac
export PYTHONPATH=.
LOG=results/rerun_queue15.log
say () { printf '[%(%Y-%m-%d %H:%M)T] %s\n' -1 "$*" >> "$LOG"; }
snap () { python -c "from huggingface_hub import snapshot_download as s; print(s('$1'))" 2>/dev/null | tail -1; }
rows () { [ -f "$1" ] && grep -c . "$1" || echo 0; }

until grep -q "P2-N qwen35_2b concept steer exit" "$LOG"; do sleep 60; done
powershell -NoProfile -c "Stop-ScheduledTask caliper-queue15; Get-CimInstance Win32_Process -Filter \"Name='bash.exe'\" | Where-Object { \$_.CommandLine -like '*rerun_queue15.sh*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }; Start-Sleep 5; Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | Where-Object { \$_.CommandLine -like '*kaggle_s3_positive_control*p2n*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
sleep 10
say "queue 15b takes over (forced runs without health checks)"

p2n () {  # p2n <tag> <hub id>
  local tag=$1 path
  path=$(snap "$2")
  [ -d "$path" ] || { say "P2-N $tag: download of $2 failed"; return; }
  local base=(--model qwen3b --model-path "$path" --quant none --compute-dtype fp32 --alpha-frac 0 0.25 0.5 1.0)
  for spec in "concept|--vector-pos concept|concept" "tail|--vector-pos template-tail|tail" \
              "sentence|--vector-recipe aperture|sentence_STAGE_aperture" "random|--control random|random_STAGE_random"; do
    IFS='|' read -r arm flag stem <<< "$spec"
    for stage in steer forced; do
      [ "$arm" = random ] && [ $stage = steer ] && continue
      local f=results/p2n/s2_${tag}_${stem//_STAGE/_$stage}
      [ "$stem" = "$arm" ] && f=results/p2n/s2_${tag}_${arm}_${stage}
      local want=120 extra=()
      [ $stage = forced ] && want=240 extra=(--no-health)
      if [ "$(rows ${f}_none.jsonl)" -ge $want ]; then say "P2-N $tag $arm $stage complete, skipped"; continue; fi
      say "starting P2-N $tag $arm $stage"
      python experiments/kaggle_s3_positive_control.py "${base[@]}" --stage $stage $flag "${extra[@]}" \
        --out results/p2n/s2_${tag}_${arm}.jsonl >> results/p2n/${tag}.log 2>&1
      say "P2-N $tag $arm $stage exit $?"
    done
  done
}

p2n qwen35_2b Qwen/Qwen3.5-2B
p2n qwen35_08b Qwen/Qwen3.5-0.8B
say "P2-N done"

path=$(snap meta-llama/Llama-3.2-1B-Instruct)
say "starting P2-H-1B ($path)"
python experiments/kaggle_p2h_hahami.py --model-path "$path" --quant none --compute-dtype fp32 --layers 0 10 \
  --alphas 5 3 1 2 4 8 12 --no-fidelity --out results/p2h1b/p2h1b_llama1b.jsonl >> results/p2h1b/p2h1b.log 2>&1
say "P2-H-1B exit $?"
say "queue 15 done"
