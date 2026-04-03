FROM python:3.11-slim

# libgomp1 is required by XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Install dependencies as a cached layer before copying source code
COPY pyproject.toml .
RUN uv pip install --system --no-cache .

# Copy only what the app needs at runtime
COPY config.py main.py ./
COPY app/ app/
COPY frontend/ frontend/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]