#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
python3 -m src.benchmark --config config/benchmark.yaml
