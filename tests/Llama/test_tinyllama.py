import os
import sys
import argparse
import copy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding, LlamaModel


def print_kv_cache(step, past_key_values, tag):
    if past_key_values is None:
        print(f"[{tag}] step={step} kv_cache=None")
        return

    # Handle both StaticCache/DynamicCache objects and plain tuple-of-tuples
    if hasattr(past_key_values, 'key_cache'):
        # HF Cache object (StaticCache, DynamicCache, etc.)
        key_cache = past_key_values.key_cache
        val_cache = past_key_values.value_cache
        num_layers = len(key_cache)
        print(f"[{tag}] step={step} num_layers={num_layers} type={type(past_key_values).__name__}")
        for layer_idx, (k, v) in enumerate(zip(key_cache, val_cache)):
            print(
                f"[{tag}] step={step} layer={layer_idx} "
                f"k_ptr={k.data_ptr()} v_ptr={v.data_ptr()} "
                f"k_shape={tuple(k.shape)} v_shape={tuple(v.shape)} "
            )
    else:
        # Legacy tuple-of-tuples format
        num_layers = len(past_key_values)
        print(f"[{tag}] step={step} num_layers={num_layers} type=tuple")
        for layer_idx, (k, v) in enumerate(past_key_values):
            print(
                f"[{tag}] step={step} layer={layer_idx} "
                f"k_ptr={k.data_ptr()} v_ptr={v.data_ptr()} "
                f"k_shape={tuple(k.shape)} v_shape={tuple(v.shape)} "
            )

@torch.no_grad()
def run_tinyllama_gen(
    device,
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    prompt="Hello!",
    dtype="float32",
    rtol=1e-3,
    atol=1e-3,
    max_new_tokens=5,
    cpu_only=False,
):
    torch.manual_seed(0)
    print("\n[Running TinyLlama-1.1B HF Test]")
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(dtype, torch.float32)

    print(f"Loading tokenizer/model from HF: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    cpu_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, device_map=None).eval()

    print("Running CPU-only generation")
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids_cpu = inputs["input_ids"]
    attention_mask_cpu = inputs["attention_mask"]
    gen_ids_cpu = input_ids_cpu
    gen_mask_cpu = attention_mask_cpu
    
    # Pre-allocate static KV cache
    # max_cache_len = input_ids_cpu.shape[1] + max_new_tokens  # prompt + generation length
    # past_key_values = StaticCache(
    #     config=cpu_model.config,
    #     max_batch_size=1,
    #     max_cache_len=max_cache_len,
    #     device=torch.device("cpu"),
    #     dtype=torch.float32,
    # )
    # for step in range(max_new_tokens):
    #     if step == 0:
    #         step_input_ids = gen_ids_cpu
    #     else:
    #         step_input_ids = gen_ids_cpu[:, -1:]
        
    #     with torch.autocast(device_type=device.type, enabled=False):
    #         out = cpu_model.forward(
    #             input_ids=step_input_ids,
    #             attention_mask=gen_mask_cpu,
    #             use_cache=True,
    #             past_key_values=past_key_values,
    #             return_dict=True,
    #         )
    #     next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    #     gen_ids_cpu = torch.cat([gen_ids_cpu, next_token], dim=1)
    #     gen_mask_cpu = torch.cat([gen_mask_cpu, torch.ones_like(next_token)], dim=1)
    #     # past_key_values = out.past_key_values
    #     print_kv_cache(step, past_key_values, "cpu")
    #     print(f"Step {step}: outputs={tokenizer.decode(gen_ids_cpu[0], skip_special_tokens=True)}")
    #     if next_token.item() == tokenizer.eos_token_id:
    #         print("[CPU] EOS reached, stopping early.")
    #         break
    # if cpu_only:
    #     return

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids_cpu = inputs["input_ids"]
    attention_mask_cpu = inputs["attention_mask"]
    input_ids_dev = input_ids_cpu.to(device)
    attention_mask_dev = attention_mask_cpu.to(device)
    # inputs_dev = inputs.to(device)
    # print("inputs: \n", inputs_dev)

    dev_model = cpu_model.to(device=device, dtype=torch_dtype)

    trace_dir = os.environ.get("TOGSIM_SSD_TRACE_DIR")
    trace_name = os.environ.get("TOGSIM_SSD_TRACE_NAME")
    if not trace_dir or not trace_name:
        raise RuntimeError("TOGSIM_SSD_TRACE_DIR and TOGSIM_SSD_TRACE_NAME must be set")

    out_dir = os.path.join(trace_dir, trace_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "model_weight_ranges.txt")
    with open(out_path, "w") as f:
        for name, p in dev_model.named_parameters():
            if p is None:
                continue
            base = p.data_ptr()
            size_bytes = p.untyped_storage().size()
            end = base + size_bytes
            f.write(
                f"{name}\tbase={base}\tend={end}\tsize_bytes={size_bytes}"
                f"\tshape={tuple(p.shape)}\tdtype={p.dtype}\n"
            )

    print("Compiling TinyLlama with torch.compile(...)")
    dev_model.forward = torch.compile(dev_model.forward, dynamic=False)

    print("Generating on NPU...")
    gen_ids = input_ids_dev
    gen_mask = attention_mask_dev
    prompt_len = input_ids_dev.shape[1]
    past_key_values = None
    # npu_past_key_values = StaticCache(
    #     config=dev_model.config,
    #     max_batch_size=1,
    #     max_cache_len=max_cache_len,
    #     device=device,
    #     dtype=torch_dtype,
    # )

    for step in range(max_new_tokens):
        if step == 0:
            step_input_ids = gen_ids
            # cache_position = torch.arange(0, prompt_len, device=device)
        else:
            step_input_ids = gen_ids[:, -1:]
            # cache_position = torch.tensor([prompt_len + step - 1], device=device)

        out = dev_model.forward(
            input_ids=step_input_ids,
            attention_mask=gen_mask,
            use_cache=True,
            past_key_values=past_key_values,
            # cache_position=cache_position,
            return_dict=True,
        )
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        gen_ids = torch.cat([gen_ids, next_token], dim=1)
        gen_mask = torch.cat([gen_mask, torch.ones_like(next_token)], dim=1)
        past_key_values = out.past_key_values
        print_kv_cache(step, past_key_values, "npu")
        print(f"Step {step}: outputs={tokenizer.decode(gen_ids[0], skip_special_tokens=True)}")
        if next_token.item() == tokenizer.eos_token_id:
            print("[NPU] EOS reached, stopping early.")
            break
    # print(tokenizer.decode(gen_ids[0], skip_special_tokens=True))


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

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids_cpu = inputs["input_ids"]
    attention_mask_cpu = inputs["attention_mask"]

    if cpu_only:
        print("Running CPU forward...")
        out_cpu = cpu_model(
            input_ids=input_ids_cpu, 
            attention_mask=attention_mask_cpu,
            use_cache=True,
            return_dict=True,
        )
        pkv = out_cpu.past_key_values
        print("num_layers:", len(pkv))
        k0, v0 = pkv[0]
        print("k0 device:", k0.device, "shape:", tuple(k0.shape), "ptr:", k0.data_ptr())
        print("v0 device:", v0.device, "shape:", tuple(v0.shape), "ptr:", v0.data_ptr())
        return

    dev_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, device_map=None).eval()
    dev_model.to(device=device, dtype=torch_dtype)
    print(f"model dtype: {next(dev_model.parameters()).dtype}")

    trace_dir = os.environ.get("TOGSIM_SSD_TRACE_DIR")
    trace_name = os.environ.get("TOGSIM_SSD_TRACE_NAME")
    if not trace_dir or not trace_name:
        raise RuntimeError("TOGSIM_SSD_TRACE_DIR and TOGSIM_SSD_TRACE_NAME must be set")

    out_dir = os.path.join(trace_dir, trace_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "model_weight_ranges.txt")
    with open(out_path, "w") as f:
        for name, p in dev_model.named_parameters():
            if p is None:
                continue
            base = p.data_ptr()
            size_bytes = p.untyped_storage().size()
            end = base + size_bytes
            f.write(
                f"{name}\tbase={base}\tend={end}\tsize_bytes={size_bytes}"
                f"\tshape={tuple(p.shape)}\tdtype={p.dtype}\n"
            ) 

    input_ids_dev = input_ids_cpu.to(device)
    attention_mask_dev = attention_mask_cpu.to(device)

    print("Compiling TinyLlama with torch.compile(...)")
    dev_model.forward = torch.compile(dev_model.forward, dynamic=False)

    print("Running CPU forward...")
    out_cpu = cpu_model.forward(
        input_ids=input_ids_cpu, 
        attention_mask=attention_mask_cpu,
        use_cache=True,
        return_dict=True,
    )

    print("Running NPU forward...")
    out_dev = dev_model.forward(
        input_ids=input_ids_dev, 
        attention_mask=attention_mask_dev,
        use_cache=True,
        return_dict=True,
    )

    # Check CPU vs NPU KV cache
    pkv = out_cpu.past_key_values
    print("num_layers:", len(pkv))
    k0, v0 = pkv[0]
    print("k0 device:", k0.device, "shape:", tuple(k0.shape), "ptr:", k0.data_ptr())
    print("v0 device:", v0.device, "shape:", tuple(v0.shape), "ptr:", v0.data_ptr())

    pkv = out_dev.past_key_values
    print("num_layers:", len(pkv))
    k0, v0 = pkv[0]
    print("k0 device:", k0.device, "shape:", tuple(k0.shape), "ptr:", k0.data_ptr())
    print("v0 device:", v0.device, "shape:", tuple(v0.shape), "ptr:", v0.data_ptr())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Custom Llama (random weights, no tokenizer)")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--max_new_tokens", type=int, default=4)
    parser.add_argument("--hf_model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--prompt", type=str, default="How are you?\nI am")
    parser.add_argument("--cpu_only", action="store_true")
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()

    sys.path.append(os.environ.get("PYTORCHSIM_ROOT_PATH", "/workspace/PyTorchSim"))
    device = torch.device("cpu" if args.cpu_only else "npu:0")
    #test_triu(device, size=(32, 128), diagonal=1)
    # if not args.cpu_only:
    #     torch.compiler.is_compiling = lambda: True # FIXME. How to fix this?

    if args.generate:
        run_tinyllama_gen(
            device=device,
            model_id=args.hf_model,
            prompt=args.prompt,
            dtype=args.dtype,
            rtol=args.rtol,
            atol=args.atol,
            max_new_tokens=args.max_new_tokens,
            cpu_only=args.cpu_only,
        )
    else:
        run_tinyllama_test(
            device=device,
            model_id=args.hf_model,
            prompt=args.prompt,
            dtype=args.dtype,
            rtol=args.rtol,
            atol=args.atol,
            cpu_only=args.cpu_only,
        )
