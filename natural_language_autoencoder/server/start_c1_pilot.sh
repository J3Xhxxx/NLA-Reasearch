#!/usr/bin/env bash
set -euo pipefail

CODE=/root/autodl-tmp/nla_compare
RESULTS=/root/autodl-tmp/results
PIDFILE="$RESULTS/c1_pilot_launcher.pid"
LAUNCH_LOG="$RESULTS/c1_pilot_launcher.log"

if [[ -f "$PIDFILE" ]]; then
  prior_pid="$(tr -cd '0-9' < "$PIDFILE")"
  if [[ -n "$prior_pid" ]] && kill -0 "$prior_pid" 2>/dev/null; then
    printf 'C1_PILOT_ALREADY_RUNNING pid=%s\n' "$prior_pid"
    exit 1
  fi
fi

nohup bash "$CODE/launch_c1_pilot.sh" > "$LAUNCH_LOG" 2>&1 < /dev/null &
launcher_pid=$!
printf '%s\n' "$launcher_pid" > "$PIDFILE"
printf 'C1_PILOT_LAUNCHED pid=%s\n' "$launcher_pid"
