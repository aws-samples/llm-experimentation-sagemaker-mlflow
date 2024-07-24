## LLM experimentation at scale using Amazon SageMaker Pipelines and MLflow

**Solution overview**

Running hundreds of experiments, comparing the results, and keeping a track of the ML lifecycle can become very complex. This is where MLflow can help streamline the ML lifecycle, from data preparation to model deployment. By integrating MLflow into your LLM workflow, you can efficiently manage experiment tracking, model versioning, and deployment, providing reproducibility. With MLflow, you can track and compare the performance of multiple LLM experiments, identify the best-performing models, and deploy them to production environments with confidence. 

You can create workflows with SageMaker Pipelines that enable you to prepare data, fine-tune models, and evaluate model performance with simple Python code for each step. 

Now you can use SageMaker managed MLflow to run LLM fine-tuning and evaluation experiments at scale. Specifically:

- MLflow can manage tracking of fine-tuning experiments, comparing evaluation results of different runs, model versioning, deployment, and configuration (such as data and hyperparameters)
- SageMaker Pipelines can orchestrate multiple experiments based on the experiment configuration 
  

The following figure shows the overview of the solution.
![](./ml-16670-arch-with-mlflow.png)

## Prerequisites 
Before you begin, make sure you have the following prerequisites in place:

- Hugging Face login token – You need a Hugging Face login token to access the models and datasets used in this post. For instructions to generate a token, see User access tokens. 
  
- SageMaker access with required IAM permissions – You need to have access to SageMaker with the necessary AWS Identity and Access Management (IAM) permissions to create and manage resources. Make sure you have the required permissions to create notebooks, deploy models, and perform other tasks outlined in this post. To get started, see Quick setup to Amazon SageMaker. Please follow this post to make sure you have proper IAM role confugured for MLflow.

## Configuring the solution

For detail instructions you can follow [this](https://aws.amazon.com/blogs/machine-learning/llm-experimentation-at-scale-using-amazon-sagemaker-pipelines-and-mlflow/) blog post 

You first need to update `ImageUri` in `config.yaml`. YOu can execute following code to get correct image uri for your region 

```python
sagemaker.image_uris.get_base_python_image_uri('<region>', py_version='310')
```
You can start with `llm_fine_tuning_experiments_mlflow.ipynb` notebook.

You also need to update `mlflow_arn` in the notebook. You can get the Tracking Server ARN when you setup the server from SageMaker Studio. 

Python functions for each step is defined in the `steps` folder. 

To download the model for fine tuning, you need to provide HuggingFace token in `finetune_llama8b_hf.py`

Once all the changes are done you can create the SageMaker Pipeline executing following code in the notebook 

```python
pipeline = Pipeline(
    name=pipeline_name,
    steps=[evaluate_finetuned_llama7b_instruction_mlflow],
    parameters=[lora_config],
)
```

## Cleanup
In order to not incur ongoing costs, delete the resources you created as part of this post:

1.	Delete the MLflow tracking server. 
2.	Run the last cell in the notebook to delete the SageMaker pipeline:

```python
sagemaker_client = boto3.client('sagemaker')
response = sagemaker_client.delete_pipeline(
    PipelineName=pipeline_name,
)

```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

