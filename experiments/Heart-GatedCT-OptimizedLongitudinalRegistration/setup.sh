#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Clone uniGradICON (feat-add-finetuning branch required for finetuning support)
if [ ! -d "uniGradICON" ]; then
    git clone -b feat-add-finetuning https://github.com/uncbiag/uniGradICON.git
else
    echo "uniGradICON/ already exists, skipping clone."
fi

# Create venv if needed
if [ ! -d "venv" ]; then
    python -m venv venv
fi

# Detect venv Python path (Windows vs Linux/Mac)
PYTHON=""
for PYTHON_CANDIDATE in \
    "venv/Scripts/python.exe" \
    "venv/Scripts/python" \
    "venv/bin/python"; do
    if [ -f "$PYTHON_CANDIDATE" ]; then
        PYTHON="$PYTHON_CANDIDATE"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "Python not found in venv, exiting."
    exit 1
fi

# Install all dependencies (including editable monai_physio and uniGradICON)
"$PYTHON" -m pip install uv
"$PYTHON" -m uv pip install -e ".[dev,docs,test,cuda13]"
