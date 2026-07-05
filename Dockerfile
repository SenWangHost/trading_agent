FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies first (cached layer — only re-runs when pyproject.toml/uv.lock change)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy source
COPY agents/       agents/
COPY broker/       broker/
COPY scanner/      scanner/
COPY graph.py      graph.py
COPY main.py       main.py
COPY polygon_client.py polygon_client.py
COPY scheduler.py  scheduler.py
COPY state.py      state.py

# Logs directory (mounted as PVC in K8s; used as-is in local runs)
RUN mkdir -p logs/reports logs/decisions

# Run as non-root
RUN adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app
USER appuser

ENTRYPOINT ["uv", "run", "python", "main.py"]
CMD ["once"]
