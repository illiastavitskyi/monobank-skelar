FROM python:3.12-slim

WORKDIR /app

# Install uv
RUN pip install uv --no-cache-dir

# Copy dependency files first (layer cache)
COPY pyproject.toml uv.lock ./

# Install dependencies (add fastapi, uvicorn, python-multipart)
RUN uv sync --no-dev --frozen

# Copy source (includes your HTML file and backend)
COPY . .

# Expose port (FastAPI default)
EXPOSE 8000

# Run your FastAPI app (not Gradio)
CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]