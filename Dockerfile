FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY autoclose ./autoclose
COPY data ./data
RUN useradd --create-home app && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')" || exit 1
# One worker: paused cases live in process memory until a Postgres checkpointer is added.
CMD ["uvicorn", "autoclose.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
