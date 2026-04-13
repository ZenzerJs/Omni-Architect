## Omni-Architect Backend (FastAPI)

### Security
- Set `OMNI_PASSWORD` in `.env` to require login.
- Set `OMNI_JWT_SECRET` in `.env` for a persistent auth cookie secret.

### DeepSeek-VL (local only, not in git)
Clone the upstream repo next to this project (or into `DeepSeek-VL/`), then install the package in editable mode:

```powershell
git clone https://github.com/deepseek-ai/DeepSeek-VL.git DeepSeek-VL
& .\venv\Scripts\pip.exe install -e .\DeepSeek-VL
```

Keep `GROQ_API_KEY` and other secrets in repo-root `.env` only (see `.env.example`). Never commit `.env`, SSH keys, or API keys.

### Run
From repo root:

```powershell
& .\venv\Scripts\python.exe -m pip install -r .\backend\requirements.txt
& .\venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Backend will store session artifacts under `specs/sessions/<session_id>/`.

