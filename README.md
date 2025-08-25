# BioBERT NER on SageMaker (Real‑Time Endpoint)

**Goal:** Fine‑tune a biomedical PLM (e.g., BioBERT) for NER, package inference, and deploy to an AWS SageMaker real‑time endpoint.

## Repo layout
```
biobert-ner-sagemaker/
├── docker/
│   └── Dockerfile
├── inference/
│   ├── predictor.py
│   └── serve.py
├── training/
│   └── train_ner.py
├── deploy_sagemaker.py
├── requirements.txt
└── README.md
```

## Quickstart
1. (Optional) Fine‑tune BioBERT with LoRA/PEFT on a free GPU (Colab/Studio Lab). Save adapter weights to `training/output/`.
2. Build the Docker image and push to ECR:
   ```bash
   cd docker
   bash ../scripts/build_and_push.sh
   ```
3. Deploy with SageMaker:
   ```bash
   python deploy_sagemaker.py --image <ecr-image-uri> --model_data s3://<bucket>/biobert/model.tar.gz
   ```
4. Invoke endpoint:
   ```bash
   python inference/predictor.py --endpoint-name <name> --text "Aspirin treats headaches"
   ```

## Notes
- Keep the endpoint off when not demonstrating to minimize cost.
- CPU instances are sufficient for small demo traffic.

## Training (HF + optional LoRA)
```bash
python training/train_ner.py --model dmis-lab/biobert-v1.1 --train training/data/sample.conll --valid training/data/sample.conll --epochs 1 --use_lora
```
