#!/usr/bin/env bash
set -e

SESSION="${SESSION:-internvl_server}"

tmux has-session -t "$SESSION" 2>/dev/null && {
  echo "tmux session '$SESSION' already exists."
  exit 0
}

tmux new -s "$SESSION" -d "bash -lc 'cd $(pwd) && source .venv/bin/activate && ./model/server/run_server.sh'"
echo "Started tmux session: $SESSION"
echo "Attach: tmux attach -t $SESSION"
