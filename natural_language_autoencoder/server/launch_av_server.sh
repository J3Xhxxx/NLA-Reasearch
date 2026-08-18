#!/usr/bin/env bash
# Serve the Gemma-3-12B NLA AV (activation verbalizer) with SGLang, in a tmux
# session, ready for 03_run_nla.py to POST input_embeds.
#
#   bash launch_av_server.sh                 # start in tmux session 'nla-av'
#   tmux attach -t nla-av                     # watch it
#   curl -fsS localhost:30000/get_model_info  # health check once it's up
#
# Notes:
#   --disable-radix-cache is REQUIRED: radix cache keys on token IDs, but
#   input_embeds requests carry none, so different vectors would alias to one
#   cache entry and silently return garbage (see nla_inference.py header).
#   AV checkpoint is served from a LOCAL dir, so no HF token is needed.
set -euo pipefail
export PATH=/root/miniconda3/bin:$PATH
export HF_HOME=${HF_HOME:-/root/autodl-tmp/hf}

AV_DIR=${AV_DIR:-/root/autodl-tmp/models/nla-gemma3-12b-L32-av}
PORT=${PORT:-30000}
MEM_FRACTION=${MEM_FRACTION:-0.85}
CTX=${CTX:-512}
LOG=${LOG:-/root/autodl-tmp/nla_av_server.log}

test -f "$AV_DIR/nla_meta.yaml" || { echo "AV checkpoint not found at $AV_DIR"; exit 1; }

CMD="python -m sglang.launch_server \
  --model-path $AV_DIR \
  --port $PORT \
  --disable-radix-cache \
  --mem-fraction-static $MEM_FRACTION \
  --context-length $CTX \
  --trust-remote-code"

echo "Launching AV server in tmux 'nla-av' (log: $LOG)"
echo "  $CMD"
tmux kill-session -t nla-av 2>/dev/null || true
tmux new-session -d -s nla-av "cd $(dirname "$AV_DIR") && export PATH=/root/miniconda3/bin:\$PATH && $CMD > $LOG 2>&1"
echo "Started. Wait ~1-2 min, then: curl -fsS localhost:$PORT/get_model_info"
echo "Tail log:  tail -f $LOG"
