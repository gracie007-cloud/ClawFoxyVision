#!/usr/bin/env bash
set -euo pipefail
EXAMPLE=${1:-lstm_example}

echo "Profiling $EXAMPLE ..."

time hyperfine --warmup 3 "cargo flamegraph --example $EXAMPLE --release" | cat 