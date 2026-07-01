"""
CPU-only token generation using the real Mixtral-8x7B-v0.1 pretrained weights
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


DEFAULT_MODEL_ID = "mistralai/Mixtral-8x7B-v0.1"
DEFAULT_PROMPT = "Once upon a time in a land far away,"

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def load_model(model_id: str, dtype: torch.dtype = torch.float32):
    print(f"Loading tokenizer from {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"Loading model onto CPU (this may take several minutes for 8x7B) ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s  (dtype={dtype})")
    return tokenizer, model


@torch.no_grad()
def generate(
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.0,
    top_k: int = 1,
    top_p: float = 0.0,
    dtype: str = "float32"
):
    torch.manual_seed(0)
    torch_dtype = DTYPE_MAP[dtype]
    tokenizer, model = load_model(args.model_id, dtype=torch_dtype)

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    print(f"\nPrompt ({input_ids.shape[1]} tokens): {prompt!r}")
    print("Generating...")

    # t0 = time.time()
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
    )
    # elapsed = time.time() - t0

    new_ids = output_ids[0, input_ids.shape[1]:]
    generated_text = tokenizer.decode(new_ids, skip_special_tokens=True)

    # n_tokens = new_ids.shape[0]
    # print(f"\nGenerated {n_tokens} tokens in {elapsed:.2f}s "
    #       f"({n_tokens / elapsed:.2f} tok/s)")
    print(f"\nOutput:\n{generated_text}")



def parse_args():
    parser = argparse.ArgumentParser(description="Mixtral-8x7B CPU token generation")
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID,
                        help="HuggingFace model ID or local path")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max_new_tokens", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=1, help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        dtype=args.dtype,
    )
