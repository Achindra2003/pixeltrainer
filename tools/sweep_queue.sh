# Modular, resumable job queue for tasks 8.3-8.7. Run: bash tools/sweep_queue.sh
# One job at a time (14 GB RAM; an agent run takes ~1.3 GB), BLAS/torch capped at 2 threads so the laptop
# stays usable. Each job writes runs/jobs/<name>.txt only when it finishes (temp file, then rename), and
# finished jobs are skipped, so stopping at any point loses only the job in progress. Delete a job's file
# to rerun it.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
PY=.venv/Scripts/python.exe
mkdir -p runs/jobs

job() {  # job <name> <command...>
  local out="runs/jobs/$1.txt"; shift
  [ -s "$out" ] && return
  echo "$(date +%H:%M) start $out"
  if "$PY" "$@" > "$out.tmp" 2> "$out.err"; then mv "$out.tmp" "$out"; rm -f "$out.err"
  else echo "FAILED $out (see $out.err)"; fi
}

for L in 0 1 2 3 4; do job "capture_L$L" evaluation/eval_capture.py --n 200 --levels $L; done
job restore_table evaluation/eval_restore.py --n 30
for L in 0 1 2 3 4; do job "register_L$L" evaluation/eval_register_degraded.py --n 200 --levels $L; done
# 8.7 sweep: 3 seeds at levels 0-2; levels 3-4 lose the screen on 90-100% of frames, so 1 short seed shows it.
for S in 0 1 2; do
  for L in 0 1 2; do job "agent_sim${L}_s$S" evaluation/run_agent.py --seeds $S --max 600 --input sim:$L; done
  for L in 1 2; do job "agent_sim${L}_norestore_s$S" evaluation/run_agent.py --seeds $S --max 600 --input sim:$L --no-restore; done
done
for L in 3 4; do
  job "agent_sim${L}_s0" evaluation/run_agent.py --seeds 0 --max 200 --input sim:$L
  job "agent_sim${L}_norestore_s0" evaluation/run_agent.py --seeds 0 --max 200 --input sim:$L --no-restore
done
echo "$(date +%H:%M) QUEUE DONE"
