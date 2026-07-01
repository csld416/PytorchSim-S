"""
CPU-only token generation using the real GPT-NeoX-20B pretrained weights
loaded from HuggingFace transformers.

Usage:
    python generate.py [--model_id <hf_model_id>] [--prompt <text>]
                       [--max_new_tokens <n>] [--temperature <t>]
                       [--top_k <k>] [--top_p <p>]
"""

import argparse
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


DEFAULT_MODEL_ID = "EleutherAI/gpt-neox-20b"
DEFAULT_PROMPT = "Once upon a time in a land far away,"

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def load_model(model_id: str, dtype: torch.dtype = torch.float32, device: str = "cpu"):
    print(f"Loading tokenizer from {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"Loading model onto CPU (this may take several minutes for 20B) ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
    ).eval()
    print(f"Model loaded in {time.time() - t0:.1f}s  (dtype={dtype})")

    if device != "cpu":
        print(f"Moving model to {device} ...")
        model = model.to(dtype=dtype, device=torch.device(device))

    return tokenizer, model


@torch.no_grad()
def generate(
    prompt: str,
    model_id: str = DEFAULT_MODEL_ID,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 1,
    top_p: float = 0.0,
    dtype: str = "float32",
    device: str = "cpu",
):
    torch.manual_seed(0)
    torch_dtype = DTYPE_MAP[dtype]
    print(f"Generating with model {model_id} (dtype={torch_dtype}, device={device}) ...")
    tokenizer, model = load_model(model_id, dtype=torch_dtype, device=device)

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(torch.device(device))
    print(f"\nPrompt ({input_ids.shape[1]} tokens): {prompt!r}")
    print("Generating...")

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
    )

    new_ids = output_ids[0, input_ids.shape[1]:]
    generated_text = tokenizer.decode(new_ids, skip_special_tokens=True)
    print(f"\nOutput:\n{generated_text}")


def parse_args():
    parser = argparse.ArgumentParser(description="GPT-NeoX-20B CPU token generation")
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID,
                        help="HuggingFace model ID or local path")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max_new_tokens", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu or npu:0)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(
        prompt=args.prompt,
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        dtype=args.dtype,
        device=args.device,
    )
