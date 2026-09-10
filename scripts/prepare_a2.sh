#!/usr/bin/env bash
# Prepare the GPU host for the a2 grids: LM extra, ICU tokenizer caches, tokenizer download.
set -euo pipefail
cd "$(dirname "$0")/.."
uv pip install -q -e ".[lm]"
for s in physionet2019 physionet2012; do
  [ -f "data/cache/$s/meta.json" ] || .venv/bin/python -m ehrjepa.data.tokenize build "data/meds/$s" --cache "data/cache/$s"
done
.venv/bin/python -c 'from transformers import AutoTokenizer; AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B"); print("tokenizer ok")'
.venv/bin/python -c 'import transformers, peft; print("transformers", transformers.__version__, "peft", peft.__version__)'
ls data/cache | grep physio
