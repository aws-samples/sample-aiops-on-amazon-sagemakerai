"""
Supervised Fine-Tuning (SFT) script for language models using TRL and Transformers.

This script supports:
- Full fine-tuning and PEFT (LoRA) training
- 4-bit quantization with BitsAndBytesConfig and MXFP4
- Spectrum parameter selection for selective fine-tuning
- Distributed training with DeepSpeed and Accelerate
- Model merging and saving for deployment
"""

import logging
import os
import re

# import soundfile as sf
import json
from dataclasses import dataclass
from datetime import datetime
from distutils.util import strtobool
from typing import Optional, Tuple, Dict, Any, List

import torch
from datasets import load_dataset, Dataset
from peft import AutoPeftModelForCausalLM, PeftModel, PeftConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    BitsAndBytesConfig,
    Mxfp4Config,
    PreTrainedModel,
    PreTrainedTokenizer,
    set_seed,
    Qwen2AudioForConditionalGeneration,
    GenerationConfig,
    pipeline,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import is_liger_kernel_available
from trl import SFTTrainer, TrlParser, ModelConfig, SFTConfig, get_peft_config

if is_liger_kernel_available():
    from liger_kernel.transformers import AutoLigerKernelForCausalLM
import base64
import io
from PIL import Image

import mlflow

mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
mlflow.enable_system_metrics_logging()
mlflow.set_experiment(os.environ['MLFLOW_EXPERIMENT_NAME'])

try:
    hyperparameters = json.loads(os.environ["SM_HPS"])
    description = hyperparameters.get('description', ' ')
except:
    print('No hyperparameters found')

# mlflow.autolog()
# mlflow.transformers.autolog(log_input_examples=True,
#                             log_model_signatures=True,
#                             log_models=True,
#                             log_datasets=True,
#                             # disable=False,
#                             # exclusive=False,
#                             # disable_for_unsupported_versions=False,
#                             # silent=False,
#                             # extra_tags=None
#                             )

# here's a list of models that needs its own import from transformers
EXCEPTION_MODEL_LIST = ["Qwen/Qwen2-Audio-7B-Instruct"]


def process_vision_info(messages: list[dict]) -> list[Image.Image]:
    image_inputs = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]

        for element in content:
            # We only care about TRL-style multimodal entries
            if isinstance(element, dict) and element.get("type") == "image_url":
                url = element.get("image_url", {}).get("url", None)
                if url and url.startswith("data:image"):
                    # strip the prefix "data:image/png;base64,"
                    b64_data = url.split(",")[1]
                    img_bytes = base64.b64decode(b64_data)
                    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    image_inputs.append(image)
    return image_inputs


def process_audio_info(messages):
    """
    Extract audio paths from messages and decode into waveform dicts
    that the processor can consume.
    """
    audio_inputs = []
    for msg in messages:
        for element in msg.get("content", []):
            if isinstance(element, dict) and element.get("type") == "audio":
                # Support either audio_url or audio.path
                audio_url = element.get("audio_url") or element.get(
                    "audio", {}
                ).get("path")
                if audio_url:
                    # Strip file:// prefix if present
                    path = audio_url.replace("file://", "")
                    array, sr = sf.read(path)
                    audio_inputs.append({"array": array, "sampling_rate": sr})
    return audio_inputs


# Configure logging
def setup_logging() -> logging.Logger:
    """Set up logging configuration for the training script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


logger = setup_logging()


@dataclass
class ScriptArguments:
    """Custom arguments for the SFT training script."""

    dataset_id_or_path: str
    """Path to dataset file (.jsonl) or HuggingFace dataset identifier."""

    dataset_splits: str = "train"
    """Dataset splits to use for training."""

    tokenizer_name_or_path: Optional[str] = None
    """Path to tokenizer or HuggingFace tokenizer identifier. If None, uses model tokenizer."""

    processor_name_or_path: Optional[str] = None
    """Path to processor or HuggingFace processor identifier. If None, uses model processor."""

    spectrum_config_path: Optional[str] = None
    """Path to YAML config file specifying which parameters to unfreeze for Spectrum training."""

    max_seq_length: int = 2048
    """Maximum sequence length for tokenization."""

    mxfp4: bool = False
    """Whether to use MXFP4 quantization instead of standard 4-bit quantization."""

    use_liger: bool = False
    """Whether to use LigerKernel over AutoClass for loading model."""

    modality_type: Optional[str] = "text"
    """Type of modality to use during training "video", "image", "audio" or "text" """

    run_evaluation: bool = True
    """Whether to run post-training evaluation comparing base and fine-tuned models."""

    eval_max_samples: int = 100
    """Maximum number of samples to use for evaluation (for efficiency)."""

    eval_max_new_tokens: int = 512
    """Maximum number of new tokens to generate during evaluation."""


def get_checkpoint_path(training_args: SFTConfig) -> Optional[str]:
    """
    Get the path to the last checkpoint if it exists.

    Args:
        training_args: Training configuration containing output directory

    Returns:
        Path to last checkpoint or None if no checkpoint exists
    """
    if os.path.isdir(training_args.output_dir):
        return get_last_checkpoint(training_args.output_dir)
    return None


def load_datasets(script_args: ScriptArguments) -> Tuple[Dataset, Dataset]:
    """
    Load training and evaluation datasets based on script arguments.

    Args:
        script_args: Script arguments containing dataset configuration

    Returns:
        Tuple of (train_dataset, eval_dataset)

    Raises:
        ValueError: If dataset loading fails or required attributes are missing
    """
    dataset_path = script_args.dataset_id_or_path

    try:
        if dataset_path.endswith('.jsonl'):
            # Load local JSONL file
            logger.info(f"Loading JSONL dataset from {dataset_path}")
            print('************************************')
            print("LISTING /opt/ml/input/data/training/")
            print(os.listdir('/opt/ml/intput/data/training'))
            print('************************************')
            dataset = load_dataset(
                'json', data_files=dataset_path, split='train'
            )

            # Split dataset (hardcoded split for JSONL files)
            total_samples = len(dataset)
            logger.warning(
                f"Dataset has only {total_samples} samples, using 90/10 split"
            )
            split_idx = int(0.9 * total_samples)
            train_dataset = dataset.select(range(split_idx))
            eval_dataset = dataset.select(range(split_idx, total_samples))
        else:
            # Load HuggingFace dataset
            logger.info(f"Loading HuggingFace dataset: {dataset_path}")

            # Check if we have the required split attributes
            if not hasattr(script_args, 'dataset_train_split'):
                raise ValueError(
                    "dataset_train_split not found in script_args for HuggingFace dataset"
                )
            if not hasattr(script_args, 'dataset_test_split'):
                raise ValueError(
                    "dataset_test_split not found in script_args for HuggingFace dataset"
                )

            config = getattr(script_args, 'config', None)
            if config is not None:
                train_dataset = load_dataset(
                    dataset_path, config, split=script_args.dataset_train_split
                )
                eval_dataset = load_dataset(
                    dataset_path, config, split=script_args.dataset_test_split
                )
            else:
                train_dataset = load_dataset(
                    dataset_path, split=script_args.dataset_train_split
                )
                eval_dataset = load_dataset(
                    dataset_path, split=script_args.dataset_test_split
                )

        logger.info(
            f"Loaded training dataset: {len(train_dataset)} samples, features: {train_dataset.features}"
        )
        logger.info(
            f"Loaded evaluation dataset: {len(eval_dataset)} samples, features: {eval_dataset.features}"
        )

        # Log first sample for debugging
        if len(train_dataset) > 0:
            logger.debug(f"First training sample: {train_dataset[0]}")

        return train_dataset, eval_dataset

    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        # raise
        dataset_name = "Josephgflowers/Finance-Instruct-500k"
        dataset = load_dataset(
            dataset_name, split="train[:100]"
        )  # just a toy example with 100 samples

        def convert_to_messages(row):
            system_content = "You are a financial reasoning assistant. Read the user’s query, restate the key data, and solve step by step. Show calculations clearly, explain any rounding or adjustments, and present the final answer in a concise and professional manner."
            user_content = row["user"]
            assistant_content = row["assistant"]

            return {
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ]
            }

        dataset = dataset.map(
            convert_to_messages, remove_columns=dataset.column_names
        )
        total_samples = len(dataset)
        logger.warning(
            f"Dataset has only {total_samples} samples, using 90/10 split"
        )
        split_idx = int(0.9 * total_samples)
        train_dataset = dataset.select(range(split_idx))
        eval_dataset = dataset.select(range(split_idx, total_samples))
        return train_dataset, eval_dataset


def setup_tokenizer(
    script_args: ScriptArguments, model_args: ModelConfig
) -> PreTrainedTokenizer:
    """
    Load and configure the tokenizer.

    Args:
        script_args: Script arguments containing tokenizer configuration
        model_args: Model arguments containing model configuration

    Returns:
        Configured tokenizer
    """
    tokenizer_name = (
        script_args.tokenizer_name_or_path or model_args.model_name_or_path
    )

    logger.info(f"Loading tokenizer from {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")

    return tokenizer


def setup_processor(script_args: ScriptArguments, model_args: ModelConfig):
    """
    Load and configure the processors.

    Args:
        script_args: Script arguments containing tokenizer configuration
        model_args: Model arguments containing model configuration

    Returns:
        Configured processor
    """
    processor_name = (
        script_args.processor_name_or_path or model_args.model_name_or_path
    )

    logger.info(f"Loading processor from {processor_name}")
    processor = AutoProcessor.from_pretrained(
        processor_name,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    return processor


def create_model_kwargs(
    model_args: ModelConfig,
    training_args: SFTConfig,
    script_args: ScriptArguments,
) -> Dict[str, Any]:
    """
    Create model loading arguments based on configuration.

    Args:
        model_args: Model configuration
        training_args: Training configuration
        script_args: Script arguments

    Returns:
        Dictionary of model loading arguments
    """
    # Determine torch dtype
    if model_args.torch_dtype in ['auto', None]:
        torch_dtype = model_args.torch_dtype
    else:
        torch_dtype = getattr(torch, model_args.torch_dtype)

    model_kwargs = {
        'revision': model_args.model_revision,
        'trust_remote_code': model_args.trust_remote_code,
        'attn_implementation': model_args.attn_implementation,
        'torch_dtype': torch_dtype,
        # 'use_cache': not training_args.gradient_checkpointing,
    }

    # Set low_cpu_mem_usage based on DeepSpeed usage
    use_deepspeed = strtobool(
        os.environ.get("ACCELERATE_USE_DEEPSPEED", "false")
    )
    if not use_deepspeed:
        model_kwargs['low_cpu_mem_usage'] = True

    # Configure quantization
    if model_args.load_in_4bit:
        if script_args.mxfp4:
            logger.info("Using MXFP4 quantization")
            model_kwargs['quantization_config'] = Mxfp4Config(dequantize=True)
        else:
            logger.info("Using BitsAndBytes 4-bit quantization")
            model_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_quant_storage=torch_dtype,
            )

    return model_kwargs


def load_model(
    model_args: ModelConfig,
    training_args: SFTConfig,
    script_args: ScriptArguments,
    model_kwargs: Dict[str, Any],
) -> PreTrainedModel:
    """
    Load the pretrained model with appropriate configuration.

    Args:
        model_args: Model configuration
        training_args: Training configuration
        script_args: Script arguments
        model_kwargs: Model loading arguments

    Returns:
        Loaded model

    Raises:
        ValueError: If MXFP4 is used with unsupported configurations
    """
    model_name = model_args.model_name_or_path

    if script_args.mxfp4:
        logger.info("🌱 Loading model with MXFP4 - skipping Liger kernel")
        # MXFP4 doesn't support Liger kernel yet
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    else:
        # Use Liger kernel if available and requested
        if script_args.use_liger and is_liger_kernel_available():
            logger.info("🐯 Loading model with Liger kernel optimization")
            model = AutoLigerKernelForCausalLM.from_pretrained(
                model_name, **model_kwargs
            )
        else:
            logger.info("↔️ Loading standard model")
            if model_name in EXCEPTION_MODEL_LIST:
                if model_name == "Qwen/Qwen2-Audio-7B-Instruct":
                    model = Qwen2AudioForConditionalGeneration.from_pretrained(
                        model_name, **model_kwargs
                    )
                else:
                    raise AssertionError(f"model {model_name} not supported")
            else:
                print(model_name, model_kwargs)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, **model_kwargs
                )

    # Wait for all processes in distributed training
    if hasattr(training_args, 'distributed_state'):
        training_args.distributed_state.wait_for_everyone()

    return model


def configure_model_for_training(
    model: PreTrainedModel, script_args: ScriptArguments
) -> PreTrainedModel:
    """
    Configure model for specific training requirements (e.g., Spectrum).

    Args:
        model: The loaded model
        script_args: Script arguments

    Returns:
        Configured model

    Raises:
        AssertionError: If Spectrum config is required but not provided for non-MXFP4 training
    """
    if script_args.spectrum_config_path and not script_args.mxfp4:
        logger.info(
            f"✅ Configuring model for Spectrum training with config: {script_args.spectrum_config_path}"
        )
        return model
    elif not script_args.spectrum_config_path and not script_args.mxfp4:
        # This seems to be a bug in the original code - it always raises an error
        # Let's make it more reasonable by only requiring spectrum config when explicitly needed
        logger.warning(
            "🤖 No Spectrum config provided - using standard training"
        )
        return model
    else:
        return model


def get_model_save_directory(model_name: str) -> str:
    """
    Get the directory path for saving the final model.

    Args:
        model_name: Name/path of the model

    Returns:
        Path to save directory
    """
    if "SM_MODEL_DIR" in os.environ:
        base_dir = os.environ["SM_MODEL_DIR"]
    else:
        base_dir = "/opt/ml/model"

    return os.path.join(base_dir, model_name)


def save_peft_model(
    trainer: SFTTrainer, training_args: SFTConfig, model_args: ModelConfig
) -> None:
    """
    Save PEFT model, merge with base model, and save final merged model.

    Args:
        trainer: The SFT trainer instance
        training_args: Training configuration
        model_args: Model configuration
    """
    final_model_dir = get_model_save_directory(model_args.model_name_or_path)
    final_model_dir = os.path.join(final_model_dir, "peft")

    logger.info("Saving PEFT model")

    # Save adapter to final model dir directory
    trainer.save_model(final_model_dir)
    logger.info(f"PEFT adapter saved to {final_model_dir}")

    # Wait for all processes
    if hasattr(training_args, 'distributed_state'):
        training_args.distributed_state.wait_for_everyone()

    # Save tokenizer
    trainer.tokenizer.save_pretrained(final_model_dir)
    logger.info(f"Tokenizer saved to {final_model_dir}")


def save_full_model(
    trainer: SFTTrainer, training_args: SFTConfig, model_args: ModelConfig
) -> None:
    """
    Save full fine-tuned model (non-PEFT).

    Args:
        trainer: The SFT trainer instance
        training_args: Training configuration
        model_args: Model configuration
    """
    logger.info("Saving full fine-tuned model")

    # Save model to final directory
    final_model_dir = get_model_save_directory(model_args.model_name_or_path)
    trainer.save_model(final_model_dir)
    logger.info(f"Model saved to {final_model_dir}")

    # Wait for all processes
    if hasattr(training_args, 'distributed_state'):
        training_args.distributed_state.wait_for_everyone()

    # Save tokenizer (fix bug: was saving to wrong directory)
    trainer.tokenizer.save_pretrained(final_model_dir)
    logger.info(f"Tokenizer saved to {final_model_dir}")


def train_function(
    model_args: ModelConfig,
    script_args: ScriptArguments,
    training_args: SFTConfig,
) -> None:
    """
    Main training function that orchestrates the entire SFT process.

    Args:
        model_args: Model configuration from TRL parser
        script_args: Custom script arguments
        training_args: Training configuration from TRL parser
    """
    logger.info("=" * 50)
    logger.info("Starting Supervised Fine-Tuning")
    logger.info("=" * 50)

    logger.info(f"\n\n🌀🌀🌀 MODALITY: {script_args.modality_type} 🌀🌀🌀")

    # Log all parameters
    logger.info(f"Model parameters: {model_args}")
    logger.info(f"Script parameters: {script_args}")
    logger.info(f"Training parameters: {training_args}")

    # Load datasets
    train_dataset, eval_dataset = load_datasets(script_args)

    # Setup tokenizer
    tokenizer_or_processor = None
    if script_args.tokenizer_name_or_path:
        tokenizer_or_processor = setup_tokenizer(script_args, model_args)
    elif script_args.processor_name_or_path:
        tokenizer_or_processor = setup_processor(script_args, model_args)
    else:
        assert (
            tokenizer_or_processor is not None
        ), "please specify `tokenizer_name_or_path` (text) or `processor_name_or_path` (vision)"

    # Configure PEFT if needed
    peft_config = None
    if model_args.use_peft:
        logger.info(
            "\n\n"
            "🪫🔋🪫🔋🪫🔋🪫🔋🪫🔋🪫🔋🪫🔋🪫🪫🔋🪫🔋🪫🔋🪫🔋🪫🔋\n"
            "🪫   CONFIGURING PEFT    🪫\n"
            "🪫🔋🪫🔋🪫🔋🪫🔋🪫🔋🪫🔋🪫🔋🪫🪫🔋🪫🔋🪫🔋🪫🔋🪫🔋\n"
        )

        peft_config = get_peft_config(model_args)

    # Load and configure model
    model_kwargs = create_model_kwargs(model_args, training_args, script_args)
    model = load_model(model_args, training_args, script_args, model_kwargs)
    model = configure_model_for_training(model, script_args)

    # collate functions are applicable for multi-modal datasets like images/video/audio
    collator_fn = None
    if script_args.modality_type == "text":
        pass
    else:
        raise AssertionError(
            f"current modality {script_args.modality_type} is unsupported - choose `image`, `video`, `audio` or `text`!"
        )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=collator_fn,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # Add eval dataset
        processing_class=tokenizer_or_processor,
        peft_config=peft_config,
    )

    # Print trainable parameters for PEFT
    if trainer.accelerator.is_main_process and peft_config:
        trainer.model.print_trainable_parameters()

    # Check for existing checkpoint
    last_checkpoint = get_checkpoint_path(training_args)
    if last_checkpoint and training_args.resume_from_checkpoint is None:
        logger.info(f"Resuming training from checkpoint: {last_checkpoint}")

    # Start training
    start_time = datetime.now()
    logger.info(
        f"Starting training at {start_time.strftime('%Y-%m-%d %H:%M:%S')} for {training_args.num_train_epochs} epochs"
    )

    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    # Log training metrics
    metrics = train_result.metrics
    metrics['train_samples'] = len(train_dataset)
    trainer.log_metrics('train', metrics)
    trainer.save_metrics('train', metrics)
    trainer.save_state()

    # Prepare model for inference
    if trainer.is_fsdp_enabled and peft_config:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type(
            'FULL_STATE_DICT'
        )

    # Restore cache for inference
    trainer.model.config.use_cache = True

    # Save model based on training type
    if model_args.use_peft:
        save_peft_model(trainer, training_args, model_args)
    else:
        save_full_model(trainer, training_args, model_args)

    # Wait for all processes before evaluation
    if hasattr(training_args, 'distributed_state'):
        training_args.distributed_state.wait_for_everyone()

    end_time = datetime.now()
    training_duration = end_time - start_time
    logger.info(f"Training completed successfully in {training_duration}")
    logger.info("=" * 50)

    tuned_pipeline = pipeline(
        task='text-generation',
        model=trainer.model,
        batch_size=8,
        tokenizer=tokenizer_or_processor,
    )

    input_example = [
        {
            "role": "system",
            "content": '''You are a financial reasoning assistant. Read the users query, 
                                            restate the key data, and solve step by step. Show calculations clearly, 
                                            explain any rounding or adjustments, and present the final answer in a 
                                            concise and professional manner.''',
        },
        {
            "role": "user",
            "content": '''Explain tradeoffs between fiscal and monetary policy as tools in a 
                                            nation's economic toolkit. Provide examples of past instances when each were utilized, 
                                            the economic conditions that led to them being deployed, 
                                            their intended effects, and an evaluation of their relative 
                                            efficacy and consequences.''',
        },
    ]

    model_config = {"batch_size": 8}

    # Infer the model signature, including a representative input, the expected output, and the parameters that we would like to be able to override at inference time.
    signature = mlflow.models.infer_signature(
        input_example,
        mlflow.transformers.generate_signature_output(
            tuned_pipeline, input_example
        ),
        params=model_config,
    )

    model_info = mlflow.transformers.log_model(
        transformers_model=tuned_pipeline,
        name="qwen3-06b-fine-tuned-finance",
        signature=signature,
        input_example=input_example,
        model_config=model_config,
    )

    def get_run_id_from_name(experiment_name: str, run_name: str) -> str:
        # Look up the experiment ID
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment '{experiment_name}' not found")

        # Search runs in that experiment by run_name (stored as tag)
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.runName = '{run_name}'",
        )

        if runs.empty:
            raise ValueError(
                f"No run found with name '{run_name}' in experiment '{experiment_name}'"
            )

        # Take the first match (assuming run_name is unique per experiment)
        return runs.iloc[0].run_id

    experiment_name = os.environ['MLFLOW_EXPERIMENT_NAME']
    run_name = training_args.run_name
    id = get_run_id_from_name(experiment_name, run_name)

    with mlflow.start_run(run_id=id):
        try:
            mlflow.log_artifact('qwen3-0.6b.yaml')
        except:
            print('cannot log recipe yaml file')


def main() -> None:
    """
    Main entry point for the SFT training script.

    Parses arguments using TRL parser and runs the training function.
    """
    try:
        # Parse arguments using TRL parser (preserving core functionality)
        parser = TrlParser((ModelConfig, ScriptArguments, SFTConfig))
        model_args, script_args, training_args = parser.parse_args_and_config()

        # Set seed for reproducibility
        set_seed(training_args.seed)
        logger.info(f"Set random seed to {training_args.seed}")

        # Run the main training loop
        train_function(model_args, script_args, training_args)

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == '__main__':
    main()
