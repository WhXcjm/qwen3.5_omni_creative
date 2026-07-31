# Copyright 2025 Huawei Technologies Co., Ltd

import os

import torch
from diffusers import HiDreamImagePipeline
from prompt_utils import run_inference
from transformer_patches import apply_patches
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

apply_patches()

MODEL_PATH = "HiDream-ai/HiDream-I1-Full"  # Model path for HiDream
FORTH_PATH = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # pretrained model path for tokenizer & text encoder
OUTPUT_PATH = "./infer_result"  # Output path
LORA_WEIGHTS = "./logs/pytorch_lora_weights.safetensors"  # Path for saved LoRA
DEVICE = "npu"

os.makedirs(OUTPUT_PATH, exist_ok=True)  # Create the output folder

tokenizer = PreTrainedTokenizerFast.from_pretrained(FORTH_PATH)
text_encoder = LlamaForCausalLM.from_pretrained(
    FORTH_PATH,
    output_hidden_states=True,
    output_attentions=True,
    torch_dtype=torch.bfloat16,
)

pipe = HiDreamImagePipeline.from_pretrained(
    MODEL_PATH,
    tokenizer_4=tokenizer,
    text_encoder_4=text_encoder,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)

if os.path.exists(LORA_WEIGHTS):  # Load Lora weights
    print(f"Loading LoRA weights from {LORA_WEIGHTS}")
    pipe.load_lora_weights(LORA_WEIGHTS)
else:
    print("LoRA weights not found. Using the base model")

pipe = pipe.to(DEVICE)
pipe.enable_model_cpu_offload()

run_inference(pipe, OUTPUT_PATH)
