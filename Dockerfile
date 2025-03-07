FROM ghcr.io/astral-sh/uv:0.5.5-python3.12-bookworm-slim

WORKDIR /app

COPY pyproject.toml .
COPY . .
RUN uv pip install --system -r pyproject.toml

EXPOSE 8501

RUN mkdir -p models/backup

CMD ["streamlit", "run", "app.py"]
