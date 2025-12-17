# model/

This module provides a unified inference interface for InternVL3.5.

## Server (run on ebcloud GPU machine)
```bash
export INTERNVL_MODEL_ID="OpenGVLab/InternVL3_5-4B-HF"
uvicorn model.server.app:app --host 0.0.0.0 --port 8000
