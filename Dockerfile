# Atlas - regime-aware multi-strategy trading system
#
# Multi-stage build: dependencies are installed in a builder stage so the final
# image carries no compiler toolchain.

FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip && pip install ".[data,dashboard,dev]"


# ---------------------------------------------------------------------------

FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="Atlas Trading System" \
      org.opencontainers.image.description="Regime-aware multi-strategy research and paper-trading platform" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    ATLAS_PROJECT_ROOT=/app \
    EXECUTION_MODE=dry_run

# git is used to record the commit hash on every backtest run.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY pyproject.toml README.md Makefile ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
COPY docs ./docs

RUN mkdir -p data/raw data/processed data/database reports/backtests reports/figures \
    && useradd --create-home --uid 1000 atlas \
    && chown -R atlas:atlas /app
USER atlas

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD atlas info > /dev/null || exit 1

# Default: show the configuration summary. Override to run any other command.
CMD ["atlas", "info"]
