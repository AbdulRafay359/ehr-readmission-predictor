# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Create a non-root user (required by Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    ENVIRONMENT=production \
    HOST=0.0.0.0 \
    PORT=7860 \
    LOG_LEVEL=INFO \
    MODEL_PATH=/home/user/app/models

# Set working directory
WORKDIR $HOME/app

# Copy requirements first for better Docker caching
COPY --chown=user requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code and models
COPY --chown=user app/ ./app/
COPY --chown=user models/ ./models/

# Hugging Face Spaces MUST run on port 7860
EXPOSE 7860

# Health check for Hugging Face
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run the FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]