#!/usr/bin/env python3
"""
run_api.py -- start the RagBenchmarkPlatform FastAPI backend.

Run this file inside VS Code (Run button / F5) so breakpoints in
src/api/main.py and the stage scripts work: the server runs in this
same process, no reloader in between.

The API serves the Data Setup pipeline stages (see src/api/main.py):
  GET  /health, GET /stages, POST /stages/{id}/run, GET /runs/{id}[/log]
The Streamlit UI (ui/app.py) talks to http://127.0.0.1:8000 by default.

OPENAI_API_KEY must be set in the environment (question generation,
auto-map judging, embeddings).
"""
import os
import sys

# This file lives at the repo root, so its own directory is the root.
# Inserting it makes src/ importable no matter what VS Code uses as the
# working directory when it runs this file.
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import uvicorn

HOST = "127.0.0.1"
PORT = 8000


def main():
    # reload=False on purpose: the reloader spawns a child process, and
    # VS Code breakpoints would land in the parent, not the process
    # actually serving requests. For auto-reload instead, run from a
    # terminal: uvicorn src.api.main:app --reload
    uvicorn.run("src.api.main:app", host=HOST, port=PORT, reload=False)


if __name__ == "__main__":
    main()
