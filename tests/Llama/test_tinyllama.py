import os
import sys
import argparse
import copy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding, LlamaModel

@torch.no_grad()
def run_tinyllama_test(
    device,
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    prompt="Hello!",
    dtype="float32",
    rtol=1e-3,
    atol=1e-3,
    max_new_tokens=5,
    cpu_only=False,
):
    print("\n[Running TinyLlama-1.1B HF Test]")
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(dtype, torch.float32)

    print(f"Loading tokenizer/model from HF: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    cpu_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, device_map=None).eval()
    cpu_model.to(device="cpu", dtype=torch_dtype)

    dev_model = None
    if not cpu_only:
        dev_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, device_map=None).eval()
        dev_model.to(device=device, dtype=torch_dtype)

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids_cpu = inputs["input_ids"]
    attention_mask_cpu = inputs["attention_mask"]

    if cpu_only:
        print("Running CPU forward...")
        out_cpu = cpu_model(input_ids=input_ids_cpu, attention_mask=attention_mask_cpu)
        logits_cpu = out_cpu.logits[:, -1, :]
        print(f"CPU logits shape: {tuple(logits_cpu.shape)}")
        return

    input_ids_dev = input_ids_cpu.to(device)
    attention_mask_dev = attention_mask_cpu.to(device)

    print("Compiling TinyLlama with torch.compile(...)")
    dev_model = torch.compile(dev_model, dynamic=False)

    print("Running CPU forward...")
    out_cpu = cpu_model(input_ids=input_ids_cpu, attention_mask=attention_mask_cpu)

    print("Running NPU forward...")
    out_dev = dev_model(input_ids=input_ids_dev, attention_mask=attention_mask_dev)

    # Compare only the last-token logits to keep memory reasonable.
    logits_cpu = out_cpu.logits[:, -1, :]
    logits_dev = out_dev.logits[:, -1, :]

    diff = (logits_dev.detach().cpu() - logits_cpu.detach().cpu()).abs().max().item()
    print(f"Max diff > {diff}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Custom Llama (random weights, no tokenizer)")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--hf_model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--prompt", type=str, default="How are you?")
    parser.add_argument("--cpu_only", action="store_true")
    args = parser.parse_args()

    sys.path.append(os.environ.get("PYTORCHSIM_ROOT_PATH", "/workspace/PyTorchSim"))
    device = torch.device("cpu" if args.cpu_only else "npu:0")
    #test_triu(device, size=(32, 128), diagonal=1)
    if not args.cpu_only:
        torch.compiler.is_compiling = lambda: True # FIXME. How to fix this?

    run_tinyllama_test(
        device=device,
        model_id=args.hf_model,
        prompt=args.prompt,
        dtype=args.dtype,
        rtol=args.rtol,
        atol=args.atol,
        cpu_only=args.cpu_only,
    )
