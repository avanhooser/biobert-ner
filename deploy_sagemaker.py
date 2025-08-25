# Minimal SageMaker deployment script (boto3). Fill in ECR image URI and S3 model_data if packaging artifacts.
import argparse, boto3, time, json, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--model_name', default='biobert-ner-' + str(int(time.time())))
    ap.add_argument('--endpoint_name', default='biobert-ner-endpoint')
    ap.add_argument('--instance_type', default='ml.m5.large')
    ap.add_argument('--model_data', default='')  # optional: s3://.../model.tar.gz
    args = ap.parse_args()

    sm = boto3.client('sagemaker')
    runtime = boto3.client('sagemaker-runtime')

    print("Creating model...")
    sm.create_model(
        ModelName=args.model_name,
        PrimaryContainer={
            'Image': args.image,
            **({'ModelDataUrl': args.model_data} if args.model_data else {}),
            'Environment': {'SAGEMAKER_PROGRAM': 'inference/serve.py'}
        },
        ExecutionRoleArn=os.environ.get("SAGEMAKER_EXEC_ROLE_ARN","")
    )

    cfg_name = args.model_name + "-cfg"
    epc_name = args.model_name + "-epc"
    print("Creating endpoint config...")
    sm.create_endpoint_config(
        EndpointConfigName=cfg_name,
        ProductionVariants=[{
            'VariantName':'AllTraffic',
            'ModelName':args.model_name,
            'InitialInstanceCount':1,
            'InstanceType':args.instance_type
        }]
    )

    print("Creating endpoint...")
    sm.create_endpoint(EndpointName=args.endpoint_name, EndpointConfigName=cfg_name)
    waiter = sm.get_waiter('endpoint_in_service')
    waiter.wait(EndpointName=args.endpoint_name)
    print("Endpoint in service:", args.endpoint_name)

if __name__ == "__main__":
    main()
