#!/usr/bin/env bash
set -euo pipefail
# Placeholder: fill in AWS_ACCOUNT_ID, AWS_REGION, ECR_REPO
: "${AWS_ACCOUNT_ID:?Set AWS_ACCOUNT_ID}"
: "${AWS_REGION:=us-east-1}"
: "${ECR_REPO:=biobert-ner}"
IMAGE_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:latest"
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com
docker build -t ${ECR_REPO} -f docker/Dockerfile .
docker tag ${ECR_REPO}:latest ${IMAGE_URI}
docker push ${IMAGE_URI}
echo "Pushed: ${IMAGE_URI}"
