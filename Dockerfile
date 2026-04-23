# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_VERSION=1.8.2 \
    POETRY_HOME="/opt/poetry" \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PYSETUP_PATH="/opt/pysetup" \
    VENV_PATH="/opt/pysetup/.venv"

# Add poetry to PATH
ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

# Install system dependencies including LaTeX for PDF generation and weasyprint deps
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
        git \
        texlive-latex-base \
        texlive-latex-extra \
        texlive-fonts-recommended \
        latexmk \
        # WeasyPrint dependencies
        libpango-1.0-0 \
        libpangocairo-1.0-0 \
        libgdk-pixbuf-2.0-0 \
        libffi-dev \
        shared-mime-info \
        libcairo2 \
        fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Set working directory
WORKDIR $PYSETUP_PATH

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Install dependencies with cache mount for faster rebuilds
RUN --mount=type=cache,target=/root/.cache/pypoetry \
    --mount=type=cache,target=/root/.cache/pip \
    poetry install --no-root --with dev --all-extras

# Copy application code
COPY . /app
WORKDIR /app

# Install the project itself
RUN --mount=type=cache,target=/root/.cache/pypoetry \
    --mount=type=cache,target=/root/.cache/pip \
    poetry install --only-root

# Create a non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Default command
CMD ["python", "-m", "valtron_core"]
