from steps.utils import endpoint_exists
from sagemaker.jumpstart.estimator import JumpStartEstimator
from sagemaker.huggingface import HuggingFace
from huggingface_hub import HfFolder
import mlflow
import time
import json
import boto3

def finetune_llama8b(preprocess_step_ret, train_config, lora_config, role, mlflow_arn, experiment_name,run_name, *args):

    mlflow.set_tracking_uri(mlflow_arn)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_id=preprocess_step_ret['run_id']) as run:
        
        
        model_id = train_config["model_id"]
        endpoint_name = train_config["endpoint_name"]
        instance_type = train_config["finetune_instance_type"]
        num_instances = train_config["finetune_num_instances"]
        epoch = train_config["epoch"]
        per_device_train_batch_size = train_config["per_device_train_batch_size"]

        lora_config = json.loads(lora_config)
        
        lora_r = lora_config["lora_r"]
        lora_alpha = lora_config["lora_alpha"]
        lora_dropout = lora_config["lora_dropout"]

        train_data_path = preprocess_step_ret["training_input_path"]

        training_job_name = f'huggingface-qlora-{train_config["epoch"]}-{lora_config["lora_r"]}-{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}'

        hyperparameters ={
            'model_id': model_id,                             # pre-trained model
            'dataset_path': '/opt/ml/input/data/training',    # path where sagemaker will save training dataset
            'epochs': epoch,                                      # number of training epochs
            'per_device_train_batch_size': per_device_train_batch_size,                 # batch size for training
            'lr': 2e-4,                                       # learning rate used during training
            'hf_token': "<huggingface_token>",                 # huggingface token to access llama 2
            'merge_weights': True,       # wether to merge LoRA into the model (needs more memory)
            'lora_r':lora_r,
            'lora_alpha':lora_alpha,
            'lora_dropout':lora_dropout,
            # 'lora_target_modules': lora_target_modules,
             'mlflow_arn': mlflow_arn, 
             'experiment_name': experiment_name,
             'run_id':preprocess_step_ret['run_id']

            }

        if endpoint_exists(endpoint_name):
            print("Endpoint already exists")
            training_job_name = None
        else:
            huggingface_estimator = HuggingFace(
            entry_point          = 'llama3_fine_tuning.py',      # train script
            source_dir           = 'scripts',         # directory which includes all the files needed for training
            instance_type        = instance_type,   # instances type used for the training job
            instance_count       = num_instances,                 # the number of instances used for training
            base_job_name        = training_job_name,          # the name of the training job
            role                 = role,              # Iam role used in training job to access AWS ressources, e.g. S3
            volume_size          = 300,               # the size of the EBS volume in GB
            py_version           = 'py310',           # the python version used in the training job
            hyperparameters      =  hyperparameters,  # the hyperparameters passed to the training job
            environment          = { "HUGGINGFACE_HUB_CACHE": "/tmp/.cache" }, # set env variable to cache models in /tmp
            image_uri             = f'763104351884.dkr.ecr.{boto3.session.Session().region_name}.amazonaws.com/huggingface-pytorch-training:2.1.0-transformers4.36.0-gpu-py310-cu121-ubuntu20.04',
            metric_definitions=[
                        {'Name': 'huggingface-textgeneration:loss', 'Regex': "'loss':\s*([0-9.]+)"},
                        {'Name': 'huggingface-textgeneration:epoch', 'Regex': "'epoch':\s*([0-9.]+)"},
                        {'Name': 'huggingface-textgeneration:train_loss', 'Regex': "'train_loss':\s*([0-9.]+)"},
                        ]
        )
            data = {'training': train_data_path}

            # starting the train job with our uploaded datasets as input
            huggingface_estimator.fit(data, wait=True)

            training_job_name = huggingface_estimator.latest_training_job.name

            return {"training_job_name": training_job_name, "run_id": preprocess_step_ret['run_id']}
