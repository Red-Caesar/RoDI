FROM ghcr.io/astral-sh/uv:0.5.5-python3.12-bookworm-slim

WORKDIR /app

COPY pyproject.toml .
COPY . .
RUN uv pip install --system -e .

EXPOSE 8501

RUN mkdir -p rodi/models/backup

CMD ["streamlit", "run", "rodi/app.py"]
