# NEUROEVOLVE FastAPI + PyTorch CPU Backend (Hugging Face Spaces Docker SDK)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NEUROEVOLVE_DATA_DIR=/tmp/neuroevolve_data \
    PORT=8080

# Create non-root user (required by Hugging Face Spaces)
RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements-deploy.txt /app/requirements-deploy.txt

# Install CPU-only PyTorch wheels first to keep image compact and fast
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r /app/requirements-deploy.txt

COPY --chown=user:user backend /app/backend

RUN mkdir -p /tmp/neuroevolve_data && chown -R user:user /tmp/neuroevolve_data /app

USER user
EXPOSE 8080

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
