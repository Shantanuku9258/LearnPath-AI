
# LearnPath AI — Production Container Image
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LEARNPATH_ENV=production \
    LEARNPATH_DATASET_PROFILE=dev

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy packaging files before editable installation
COPY README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir -e .

# Copy remaining application resources
COPY config/ ./config/
COPY schemas/ ./schemas/
COPY reports/ ./reports/
COPY data/ ./data/
COPY models/ ./models/

RUN python -m learnpath_ai --help

ENTRYPOINT ["python", "-m", "learnpath_ai"]
CMD ["evaluate", "compare", "--profile", "dev"]
