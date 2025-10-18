#!/bin/bash
# Fixed run script for MapAnything service

cd /home/mayor/Noesis_Devel  # Project root
export PYTHONPATH="${PYTHONPATH}:."

# Run uvicorn via python module (avoids PATH issues)
python3 -m uvicorn services.mapanything_svc.server:app --host 127.0.0.1 --port 8003
