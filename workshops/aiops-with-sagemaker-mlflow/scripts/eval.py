import os
import json

import boto3
import torch
import mlflow
import pandas as pd
from datasets import load_dataset
from transformers import pipeline
from mlflow.metrics import (
    bleu,
    rouge1,
    rouge2,
    rougeL,
    rougeLsum,
    latency,
    token_count,
)
from mlflow.metrics.genai import (
    EvaluationExample,
    answer_correctness,
    answer_relevance,
    answer_similarity,
    faithfulness,
    relevance,
    make_genai_metric,
)

try:
    hyperparameters = json.loads(os.environ["SM_HPS"])
    print("Hyperparameters:", hyperparameters)
    model_id = hyperparameters.get('model_id', None)
    adapter_path = hyperparameters.get('adapter_path', None)
    experiment_name = hyperparameters.get('experiment_name', None)
    run_name = hyperparameters.get('run_name', None)

except:
    print("No hyperparameters found")

session = boto3.Session(region_name='us-east-1')
credentials = session.get_credentials()
credentials = credentials.get_frozen_credentials()
access_key, secret_key = credentials.access_key, credentials.secret_key

if credentials.token:
    token = credentials.token
    os.environ["AWS_SESSION_TOKEN"] = token

os.environ["AWS_REGION"] = "us-east-1"
os.environ["AWS_ACCESS_KEY_ID"] = access_key
os.environ["AWS_SECRET_ACCESS_KEY"] = secret_key

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME"))
mlflow.enable_system_metrics_logging()

model_id = os.environ.get("MODEL_ID")
pipe = pipeline(
    "text-generation",
    model=model_id,
    dtype='auto',
    device_map="auto",
)
pipe.generation_config.pad_token_id = pipe.tokenizer.eos_token_id


# fine-tuned adapter

import os
print(os.getcwd())
print(os.listdir('./'))
# source = adapter_path
target = 'adapter'

os.makedirs(target, exist_ok=True)
cmd = f'aws s3 sync {adapter_path} {target}'
_ = os.system(cmd)

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_name = "Qwen/Qwen3-0.6B" # e.g., "meta-llama/Llama-2-7b-hf"
adapter_path = "./adapter/" # The path where you saved the adapter

base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tuned_model = PeftModel.from_pretrained(base_model, adapter_path)

tuned_pipe = pipeline(
    "text-generation",
    model=tuned_model,
    dtype='auto',
    device_map="auto",
    tokenizer=tokenizer
)
tuned_pipe.generation_config.pad_token_id = tuned_pipe.tokenizer.eos_token_id


# dataset = load_dataset(os.environ.get("HF_DATASET"))
# eval_df = pd.DataFrame(dataset["train"])
# eval_df = eval_df.rename(columns={"output": "ground_truth"})
# eval_df['inputs'] = eval_df['instruction'] + " " + eval_df['input']
# eval_df = eval_df[["inputs", "ground_truth"]]

dataset = load_dataset(os.environ.get("HF_DATASET"), split='train[:10]')
eval_df = pd.DataFrame(dataset)
eval_df = eval_df.rename(columns={"user": "inputs"})
eval_df = eval_df.rename(columns={"assistant": "ground_truth"})
eval_df = eval_df[["inputs", "ground_truth"]]


def predict(questions):
    responses = []
    for question in questions['inputs']:
        messages = [
            {
                "role": "system",
                "content": "ou are a financial reasoning assistant. Read the user’s query, restate the key data, and solve step by step. Show calculations clearly, explain any rounding or adjustments, and present the final answer in a concise and professional manner.",
            },
            {"role": "user", "content": question},
        ]
        output = pipe(
            messages,
            max_new_tokens=256,
        )
        responses.append(output[0]["generated_text"][-1]['content'])
    return responses

def tuned_predict(questions):
    responses = []
    for question in questions['inputs']:
        messages = [
            {
                "role": "system",
                "content": "ou are a financial reasoning assistant. Read the user’s query, restate the key data, and solve step by step. Show calculations clearly, explain any rounding or adjustments, and present the final answer in a concise and professional manner.",
            },
            {"role": "user", "content": question},
        ]
        output = tuned_pipe(
            messages,
            max_new_tokens=256,
        )
        responses.append(output[0]["generated_text"][-1]['content'])
    return responses


# Create an example to describe what answer_similarity means like for this problem.
example = EvaluationExample(
    input="What is MLflow?",
    output="MLflow is an open-source platform for managing machine "
    "learning workflows, including experiment tracking, model packaging, "
    "versioning, and deployment, simplifying the ML lifecycle.",
    score=4,
    justification="The definition effectively explains what MLflow is "
    "its purpose, and its developer. It could be more concise for a 5-score.",
    grading_context={
        "targets": "MLflow is an open-source platform for managing "
        "the end-to-end machine learning (ML) lifecycle. It was developed by Databricks, "
        "a company that specializes in big data and machine learning solutions. MLflow is "
        "designed to address the challenges that data scientists and machine learning "
        "engineers face when developing, training, and deploying machine learning models."
    },
)


judge = "bedrock:/us.anthropic.claude-3-5-haiku-20241022-v1:0"
judge_parameters = {
    "temperature": 0,
    "max_tokens": 256,
    "anthropic_version": "bedrock-2023-05-31",
}




answer_similarity_metric = answer_similarity(
    model=judge, parameters=judge_parameters, examples=[example]
)

answer_correctness_metric = answer_correctness(
    model=judge,
    parameters=judge_parameters,
)

answer_relevance_metric = answer_relevance(
    model=judge,
    parameters=judge_parameters,
)

relevance_metric = relevance(
    model=judge,
    parameters=judge_parameters,
)

faithfulness_examples = [
    EvaluationExample(
        input="How do I disable MLflow autologging?",
        output="mlflow.autolog(disable=True) will disable autologging for all functions. In Databricks, autologging is enabled by default. ",
        score=2,
        justification="The output provides a working solution, using the mlflow.autolog() function that is provided in the context.",
        grading_context={
            "context": "mlflow.autolog(log_input_examples: bool = False, log_model_signatures: bool = True, log_models: bool = True, log_datasets: bool = True, disable: bool = False, exclusive: bool = False, disable_for_unsupported_versions: bool = False, silent: bool = False, extra_tags: Optional[Dict[str, str]] = None) → None[source] Enables (or disables) and configures autologging for all supported integrations. The parameters are passed to any autologging integrations that support them. See the tracking docs for a list of supported autologging integrations. Note that framework-specific configurations set at any point will take precedence over any configurations set by this function."
        },
    ),
    EvaluationExample(
        input="How do I disable MLflow autologging?",
        output="mlflow.autolog(disable=True) will disable autologging for all functions.",
        score=5,
        justification="The output provides a solution that is using the mlflow.autolog() function that is provided in the context.",
        grading_context={
            "context": "mlflow.autolog(log_input_examples: bool = False, log_model_signatures: bool = True, log_models: bool = True, log_datasets: bool = True, disable: bool = False, exclusive: bool = False, disable_for_unsupported_versions: bool = False, silent: bool = False, extra_tags: Optional[Dict[str, str]] = None) → None[source] Enables (or disables) and configures autologging for all supported integrations. The parameters are passed to any autologging integrations that support them. See the tracking docs for a list of supported autologging integrations. Note that framework-specific configurations set at any point will take precedence over any configurations set by this function."
        },
    ),
]

faithfulness_metric = faithfulness(
    model=judge, parameters=judge_parameters, examples=faithfulness_examples
)

# Create a metric for professionalism
professionalism_metric = make_genai_metric(
    name="professionalism",
    definition=(
        "Professionalism refers to the use of a formal, respectful, and appropriate style of communication that is tailored to the context and audience. It often involves avoiding overly casual language, slang, or colloquialisms, and instead using clear, concise, and respectful language"
    ),
    grading_prompt=(
        "Professionalism: If the answer is written using a professional tone, below "
        "are the details for different scores: "
        "- Score 1: Language is extremely casual, informal, and may include slang or colloquialisms. Not suitable for professional contexts."
        "- Score 2: Language is casual but generally respectful and avoids strong informality or slang. Acceptable in some informal professional settings."
        "- Score 3: Language is balanced and avoids extreme informality or formality. Suitable for most professional contexts. "
        "- Score 4: Language is noticeably formal, respectful, and avoids casual elements. Appropriate for business or academic settings. "
        "- Score 5: Language is excessively formal, respectful, and avoids casual elements. Appropriate for the most formal settings such as textbooks. "
    ),
    examples=[
        EvaluationExample(
            input="What is MLflow?",
            output=(
                "MLflow is like your friendly neighborhood toolkit for managing your machine learning projects. It helps you track experiments, package your code and models, and collaborate with your team, making the whole ML workflow smoother. It's like your Swiss Army knife for machine learning!"
            ),
            score=2,
            justification=(
                "The response is written in a casual tone. It uses contractions, filler words such as 'like', and exclamation points, which make it sound less professional. "
            ),
        )
    ],
    version="v1",
    model=judge,
    parameters=judge_parameters,
    grading_context_columns=[],
    aggregations=["mean", "variance", "p90"],
    greater_is_better=True,
)

# Create a metric for helpfulness
helpfulness_metric = make_genai_metric(
    name="helpfulness",
    definition=(
        "Helpfulness evaluates how well the generated response directly addresses the user's query. "
        "A helpful response is accurate, relevant, clear, and provides all necessary information to solve the user's problem or answer their question."
    ),
    grading_prompt=(
        "Helpfulness: If the answer is direct, accurate, and comprehensive in response to the input, score it based on the following scale:"
        "- Score 1: The response is unhelpful. It is incorrect, irrelevant, or fails to address the user's query."
        "- Score 2: The response is somewhat helpful but lacks key information or is partially incorrect."
        "- Score 3: The response is generally helpful and correct, but it is not comprehensive or could be better organized."
        "- Score 4: The response is very helpful. It is accurate, relevant, and provides a clear and complete answer to the query."
        "- Score 5: The response is exceptionally helpful. It is accurate, comprehensive, and provides extra relevant details or context that exceed the user's expectations."
    ),
    examples=[
        EvaluationExample(
            input="How do I install MLflow?",
            output=(
                "You can easily install the MLflow library by running the following command in your terminal or command prompt: `pip install mlflow`."
            ),
            score=4,
            justification=(
                "The response is a direct and accurate answer to the user's question. It provides the exact command needed, making it highly useful."
            ),
        ),
        EvaluationExample(
            input="Explain what a neural network is.",
            output=(
                "A neural network is a computational model inspired by the human brain. It's composed of interconnected nodes (neurons) organized in layers that process and pass information to make predictions or find patterns in data. Think of it like a series of interconnected filters, each one learning to recognize a different feature of the data."
            ),
            score=5,
            justification=(
                "The response is accurate, comprehensive, and uses an analogy ('interconnected filters') to make the complex topic easier to understand, which adds significant value."
            ),
        ),
    ],
    version="v1",
    model=judge,
    parameters=judge_parameters,
    grading_context_columns=[],
    aggregations=["mean", "variance", "p90"],
    greater_is_better=True,
)


# Create a metric for politeness
politeness_metric = make_genai_metric(
    name="politeness",
    definition=(
        "Politeness measures the degree to which a response is courteous, respectful, and friendly. "
        "A polite response uses appropriate salutations, avoids dismissive language, and maintains a positive tone."
    ),
    grading_prompt=(
        "Politeness: Score the following response based on how polite, respectful, and friendly it is. "
        "Use the following scale:"
        "- Score 1: The response is rude, dismissive, or contains disrespectful language."
        "- Score 2: The response is neutral or terse, showing a lack of courtesy."
        "- Score 3: The response is generally polite but lacks warmth or friendliness."
        "- Score 4: The response is polite and friendly, using respectful language and a positive tone."
        "- Score 5: The response is exceptionally polite and warm, using thoughtful and courteous language throughout."
    ),
    examples=[
        EvaluationExample(
            input="Can you tell me how to find the nearest coffee shop?",
            output=(
                "Please provide your current location, and I will be happy to help you find the nearest coffee shop."
            ),
            score=4,
            justification=(
                "The response is polite and helpful. It uses 'Please' and 'I will be happy to help,' which contributes to a positive tone."
            ),
        ),
        EvaluationExample(
            input="I have a question about my order.",
            output=("What's the order number? Just tell me."),
            score=2,
            justification=(
                "The response is direct but terse and lacks politeness. It comes across as neutral and slightly demanding."
            ),
        ),
    ],
    version="v1",
    model=judge,
    parameters=judge_parameters,
    grading_context_columns=[],
    aggregations=["mean", "variance", "p90"],
    greater_is_better=True,
)


def get_run_id_from_name(experiment_name: str, run_name: str) -> str:
    # Look up the experiment ID
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    # Search runs in that experiment by run_name (stored as tag)
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'"
    )

    if runs.empty:
        raise ValueError(f"No run found with name '{run_name}' in experiment '{experiment_name}'")

    # Take the first match (assuming run_name is unique per experiment)
    return runs.iloc[0].run_id

parent_run_id = get_run_id_from_name(experiment_name, run_name)

with mlflow.start_run(run_id=parent_run_id):
    with mlflow.start_run(run_name='base-model-eval', nested=True):
        results = mlflow.evaluate(
            predict,
            eval_df.head(int(os.environ.get("NUM_SAMPLES", 5))),
            evaluators="default",
            targets="ground_truth",
            extra_metrics=[
                bleu(),
                rouge1(),
                rouge2(),
                rougeL(),
                rougeLsum(),
                latency(),
                token_count(),
                answer_similarity_metric,
                answer_correctness_metric,
                answer_relevance_metric,
                # relevance_metric, # requires context column
                # faithfulness_metric, # requires context column
                professionalism_metric,
                helpfulness_metric,
                politeness_metric,
            ],
        )

        try:
            mlflow.log_metrics(results.metrics)
        except:
            print('could not log metrics to mlflow')

        print(results.metrics)
        results.tables['eval_results_table'].to_csv(
            '/opt/ml/output/data/eval_results_table.csv'
        )
        # results.tables['eval_results_table'].to_csv('results.csv')
        print(results.tables['eval_results_table'])


    with mlflow.start_run(run_name='tuned-model-eval', nested=True):
        tuned_results = mlflow.evaluate(
            tuned_predict,
            eval_df.head(int(os.environ.get("NUM_SAMPLES", 5))),
            evaluators="default",
            targets="ground_truth",
            extra_metrics=[
                bleu(),
                rouge1(),
                rouge2(),
                rougeL(),
                rougeLsum(),
                latency(),
                token_count(),
                answer_similarity_metric,
                answer_correctness_metric,
                answer_relevance_metric,
                # relevance_metric, # requires context column
                # faithfulness_metric, # requires context column
                professionalism_metric,
                helpfulness_metric,
                politeness_metric,
            ],
        )

        try:
            mlflow.log_metrics(tuned_results.metrics)
        except:
            print('could not log metrics to mlflow')

        print(tuned_results.metrics)
        tuned_results.tables['eval_results_table'].to_csv(
            '/opt/ml/output/data/eval_results_table.csv'
        )
        # results.tables['eval_results_table'].to_csv('results.csv')
        print(tuned_results.tables['eval_results_table'])
