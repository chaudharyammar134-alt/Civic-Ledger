#!/usr/bin/env bash
# Installs dependencies (first run only) and starts the Civic Ledger server.
set -e
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q -r requirements.txt

cd backend
echo ""
echo "Starting server..."
python3 app.py
