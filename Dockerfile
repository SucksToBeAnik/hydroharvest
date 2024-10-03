# Base stage - common steps
FROM python:latest AS base

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Dev stage
FROM base AS dev

EXPOSE 80

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80", "--reload"]

# Prod stage
FROM base AS prod

EXPOSE 81

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "81"]
