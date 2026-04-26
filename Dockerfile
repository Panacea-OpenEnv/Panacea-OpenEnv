FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860

COPY requirements-hf.txt .
RUN pip install --no-cache-dir -r requirements-hf.txt

COPY openenv_panacea ./openenv_panacea

EXPOSE 7860

CMD ["uvicorn", "openenv_panacea.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
