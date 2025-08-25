# Minimal CPU inference container for SageMaker
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1     PORT=8080

RUN apt-get update && apt-get install -y --no-install-recommends     git curl &&     rm -rf /var/lib/apt/lists/*

WORKDIR /opt/program
COPY requirements.txt /opt/program/requirements.txt
RUN pip install --no-cache-dir -r /opt/program/requirements.txt

COPY inference /opt/program/inference
COPY training /opt/program/training
COPY inference/serve.py /usr/local/bin/serve
RUN chmod +x /usr/local/bin/serve

# SageMaker expects the container to listen on 8080 and serve /invocations and /ping
EXPOSE 8080
CMD ["/usr/local/bin/serve"]
