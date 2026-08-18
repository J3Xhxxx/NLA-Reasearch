#!/usr/bin/env bash
set -uo pipefail

manifest=/root/autodl-tmp/results/n5_model_weights_v1.sha256
log=/root/autodl-tmp/results/n5_model_weights_v1.check.log
status=/root/autodl-tmp/results/n5_model_weights_v1.check.exit

rm -f "$status"
if sha256sum -c "$manifest" >"$log" 2>&1; then
    printf '0\n' >"$status"
else
    code=$?
    printf '%s\n' "$code" >"$status"
fi
