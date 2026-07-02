FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
COPY static ./static
COPY eval ./eval
COPY benchmarks ./benchmarks
COPY docs ./docs
COPY playbooks ./playbooks
COPY workflows ./workflows
COPY *.py ./
COPY config.yaml ./

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8300

CMD ["python", "ultra_server.py", "--host", "0.0.0.0", "--port", "8300"]
