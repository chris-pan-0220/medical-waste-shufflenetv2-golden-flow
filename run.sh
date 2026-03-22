#!/usr/bin/env bash
# run.sh — 一鍵執行 ROM v4 (Split-ROM) 驗證
# 使用方式: bash run.sh [config.toml]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-${SCRIPT_DIR}/workspace/config.toml}"

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: Config file not found: $CONFIG"
    exit 1
fi

echo "=== ROM v4 (Split-ROM) Verification ==="
echo "Config : $CONFIG"
echo "========================================"

cd "${SCRIPT_DIR}/workspace"
uv run verify_romv4.py --config "$CONFIG"
