import os
from accelerate import Accelerator
from huggingface_hub import login
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sagemaker.remote_function import remote
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import transformers
from datasets import load_from_disk, load_dataset
import argparse
import bitsandbytes as bnb
from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import Dataset, DatasetDict
import mlflow
from mlflow.models import infer_signature

def parse_arge():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    # add model id and dataset path argument
    parser.add_argument(
        "--model_id",
        type=str,
        help="Model id to use for training.",
    )
    parser.add_argument(
        "--dataset_path", type=str, default="lm_dataset", help="Path to dataset."
    )
    parser.add_argument(
        "--hf_token", type=str, default="", help="Path to dataset."
    )
    # add training hyperparameters for epochs, batch size, learning rate, and seed
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--lora_r", type=int, default=8, help="Loar R"
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=16, help="Loar alpha"
    )
    parser.add_argument(
        "--lora_dropout", type=float, default=0.05, help="Loar dropout"
    )

    parser.add_argument(
        "--lora_target_modules", type=list, default="q_proj,v_proj", help="Loar target modules"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size to use for training.",
    )
    parser.add_argument(
        "--lr", type=float, default=5e-5, help="Learning rate to use for training."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed to use for training."
    )
    parser.add_argument(
        "--gradient_checkpointing",
        type=bool,
        default=True,
        help="Path to deepspeed config file.",
    )
    parser.add_argument(
        "--bf16",
        type=bool,
        default=True if torch.cuda.get_device_capability()[0] == 8 else False,
        help="Whether to use bf16.",
    )
    parser.add_argument(
        "--merge_weights",
        type=bool,
        default=True,
        help="Whether to merge LoRA weights with base model.",
    )
    parser.add_argument(
        "--mlflow_arn", type=str
    )
    parser.add_argument(
        "--experiment_name", type=str
    )
    parser.add_argument(
        "--run_id", type=str
    )
    args, _ = parser.parse_known_args()

    if args.hf_token:
        print(f"Logging into the Hugging Face Hub with token {args.hf_token[:10]}...")
        login(token=args.hf_token)

    return args

LLAMA_3_CHAT_TEMPLATE = (
    "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
            "{{ message['content'] }}"
        "{% elif message['role'] == 'user' %}"
            "{{ '\n\nHuman: ' + message['content'] +  eos_token }}"
        "{% elif message['role'] == 'assistant' %}"
            "{{ '\n\nAssistant: '  + message['content'] +  eos_token  }}"
        "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '\n\nAssistant: ' }}"
    "{% endif %}"
)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def find_all_linear_names(hf_model):
    lora_module_names = set()
    for name, module in hf_model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)

# This code is based on https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/scripts/run_fsdp_qlora.py
def train_fn(
        args,
        lora_r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        num_train_epochs=1,
        chunk_size=2048,
        gradient_checkpointing=False,
        merge_weights=False,
        token=None
):  
    print("############################################")
    print("Number of GPUs: ", torch.cuda.device_count())
    print("############################################")
    
    accelerator = Accelerator()

    model_id = args.model_id

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE
    
    if token is not None:
        login(token=token)

    data_df = pd.read_json(os.path.join(args.dataset_path, "train_dataset.json"), orient='records', lines=True)
    train, test = train_test_split(data_df, test_size=0.3)

    train_dataset = Dataset.from_pandas(train)
    test_dataset = Dataset.from_pandas(test)

    dataset = DatasetDict({"train": train_dataset, "test": test_dataset})  

    print(f"Loaded train dataset with {len(train_dataset)} samples")

    def template_dataset(examples):
        return{"text":  tokenizer.apply_chat_template(examples["messages"], tokenize=False)}
    
    train_dataset = dataset["train"].map(template_dataset, remove_columns=["messages"])
    test_dataset = dataset["test"].map(template_dataset, remove_columns=["messages"])
    
    with accelerator.main_process_first():
        lm_train_dataset = train_dataset.map(
            lambda sample: tokenizer(sample["text"]), batched=True, batch_size=per_device_train_batch_size
        )
        
    # train_dataset = train_dataset.map(template_dataset, remove_columns=["messages"])
    lm_train_dataset=lm_train_dataset.select(range(100))
    # Print total number of samples
    print(f"Total number of train samples: {len(lm_train_dataset)}")

    print(lm_train_dataset[0])

    if test_dataset is not None:
        with accelerator.main_process_first():
            lm_test_dataset = test_dataset.map(
            lambda sample: tokenizer(sample["text"]), batched=True, batch_size=per_device_train_batch_size
        )
        print(f"Total number of test samples: {len(lm_test_dataset)}")
    else:
        lm_test_dataset = None

    lm_test_dataset=lm_test_dataset.select(range(10))

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map={'':torch.cuda.current_device()},
        cache_dir="/tmp/.cache"
    )

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # get lora target modules
    modules = find_all_linear_names(model)
    print(f"Found {len(modules)} modules to quantize: {modules}")

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)

    model = model.to(accelerator.device)
    if test_dataset is not None:
        model, lm_train_dataset, lm_test_dataset = accelerator.prepare(
            model, lm_train_dataset, lm_test_dataset
        )
    else:
        model, lm_train_dataset = accelerator.prepare(
            model, lm_train_dataset
        )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=lm_train_dataset,
        eval_dataset=lm_test_dataset,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=gradient_checkpointing,
            logging_steps=2,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            bf16=True,
            save_strategy="no",
            output_dir="outputs",
            report_to="mlflow",
            run_name="llama3-peft",
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    model.config.use_cache = False

    mlflow.set_tracking_uri(args.mlflow_arn)
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_id=args.run_id) as run:
        lora_params = {'lora_alpha':lora_alpha, 'lora_dropout': lora_dropout, 'r':lora_r, 'bias': 'none', 'task_type': 'CAUSAL_LM', 'target_modules': modules}
        mlflow.log_params(lora_params)

        trainer.train()

        if merge_weights:
            output_dir = "/tmp/model"

            # merge adapter weights with base model and save
            # save int 4 model
            trainer.model.save_pretrained(output_dir, safe_serialization=False)
            # clear memory
            del model
            del trainer
            
            torch.cuda.empty_cache()

            # load PEFT model in fp16
            model = AutoPeftModelForCausalLM.from_pretrained(
                output_dir,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                cache_dir="/tmp/.cache"
            )
            
            # Merge LoRA and base model and save
            model = model.merge_and_unload()
            model.save_pretrained(
                "/opt/ml/model", safe_serialization=True, max_shard_size="2GB"
            )
        else:
            model.save_pretrained("/opt/ml/model", safe_serialization=True)

        tmp_tokenizer = AutoTokenizer.from_pretrained(model_id)
        tmp_tokenizer.save_pretrained("/opt/ml/model")
        params = {
            "top_p": 0.9,
            "temperature": 0.9,
            "max_new_tokens": 200,
        }
        signature = infer_signature("inputs","generated_text", params=params)
        mlflow.transformers.log_model(
            transformers_model={"model": model, "tokenizer": tmp_tokenizer},
            # prompt_template=prompt_template,
            signature=signature,
            artifact_path="model",  # This is a relative path to save model files within MLflow run
            model_config = params
        )

def main():
    args = parse_arge()
    train_fn(
    args,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=args.epochs,
    merge_weights=True,
    token=args.hf_token,
    lora_r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
)


if __name__ == "__main__":
    main()