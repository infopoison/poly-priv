FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr (important for Railway logs)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code (dev/ excluded via .dockerignore)
COPY src/ ./src/

# Create data directory for signal logs
RUN mkdir -p /app/data

# Create non-root user for security
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

# Health check endpoint port
EXPOSE 8080

# Entry point
CMD ["python", "-m", "src.main"]