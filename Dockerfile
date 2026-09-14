FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/app/data/models GIT_PYTHON_REFRESH=quiet
WORKDIR /app
COPY requirements.txt requirements-eval.txt ./
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt -r requirements-eval.txt
COPY . .
RUN useradd --create-home app && mkdir -p /app/data/uploads /app/data/evaluations && chown -R app:app /app
USER app
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.headless=true"]
