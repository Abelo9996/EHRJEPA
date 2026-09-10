#!/usr/bin/env bash
# Diagnose a failing tokenizer build: which parquet paths does it touch and which is malformed?
set -uo pipefail
cd "$(dirname "$0")/.."
src="$1"
echo "== files under $src"; find "$src" -type f | sort
echo "== tail bytes of each parquet"
for f in $(find "$src" -name '*.parquet' | sort); do printf '%s: ' "$f"; tail -c 4 "$f"; echo; done
echo "== traceback frames in ehrjepa"
.venv/bin/python -m ehrjepa.data.tokenize build "$src" --cache /tmp/diag_cache 2>&1 | grep -E 'ehrjepa/data|Error' | grep -v site-packages | head -n 10
