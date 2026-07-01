import os
import subprocess, sys
import argparse
import copy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding, LlamaModel

@torch.no_grad()
def run_llama_gen(
    device,
    model_id="meta-llama/Llama-2-7b-hf",
    prompt="Hello!",
    dtype="float32",
    rtol=1e-3,
    atol=1e-3,
    max_new_tokens=5,
    cpu_only=False,
):
    torch.manual_seed(0)
    print("\n[Running Llama-2-7B HF Test]")
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(dtype, torch.float32)

    print(f"Loading tokenizer/model from HF: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    cpu_model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch_dtype, 
        device_map="cpu",
        low_cpu_mem_usage=True,
    ).eval()

    print("Running CPU-only generation")
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids_cpu = inputs["input_ids"]
    attention_mask_cpu = inputs["attention_mask"]
    gen_ids_cpu = input_ids_cpu
    gen_mask_cpu = attention_mask_cpu
    
    # Pre-allocate static KV cache
    # max_cache_len = input_ids_cpu.shape[1] + max_new_tokens + 1 # prompt + generation length
    # past_key_values = StaticCache(
    #     config=cpu_model.config,
    #     max_batch_size=1,
    #     max_cache_len=max_cache_len,
    #     device=torch.device("cpu"),
    #     dtype=torch_dtype,
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
    #     # print_kv_cache(step, past_key_values, "cpu")
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
    # Run merge_weight_ranges.py
    subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(__file__), "../../merge_weight_ranges.py")],
        check=True,
    )

    print("Compiling Llama-2-7B with torch.compile(...)")
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

    # Patch TOGSimulator.launch_kernel to trace every NPU dispatch
    from Simulator.simulator import TOGSimulator
    _dispatch_log = []
    _orig_launch = TOGSimulator.launch_kernel
    def _traced_launch(self, device_index, stream_index, tog_path, attribute_path, timestamp=0):
        _dispatch_log.append(tog_path)
        print(f"  [NPU dispatch #{len(_dispatch_log)}] {tog_path}")
        return _orig_launch(self, device_index, stream_index, tog_path, attribute_path, timestamp)
    TOGSimulator.launch_kernel = _traced_launch

    for step in range(max_new_tokens):
        if step == 0:
            step_input_ids = gen_ids
            # cache_position = torch.arange(0, prompt_len, device=device)
        else:
            step_input_ids = gen_ids[:, -1:]
            # cache_position = torch.tensor([prompt_len + step - 1], device=device)

        # with TOGSimulator() as sim:
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
        # print_kv_cache(step, past_key_values, "npu")
        print(f"Step {step}: outputs={tokenizer.decode(gen_ids[0], skip_special_tokens=True)}")
        if next_token.item() == tokenizer.eos_token_id:
            print("[NPU] EOS reached, stopping early.")
            break
    # print(tokenizer.decode(gen_ids[0], skip_special_tokens=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Custom Llama (random weights, no tokenizer)")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--max_new_tokens", type=int, default=3)
    parser.add_argument("--hf_model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--prompt", type=str, default="Machine learning is a powerful tool")
    parser.add_argument("--cpu_only", action="store_true")
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()

    sys.path.append(os.environ.get("PYTORCHSIM_ROOT_PATH", "/workspace/PyTorchSim"))
    device = torch.device("cpu" if args.cpu_only else "npu:0")
    #test_triu(device, size=(32, 128), diagonal=1)
    if not args.cpu_only:
        torch.compiler.is_compiling = lambda: True # FIXME. How to fix this?

    run_llama_gen(
        device=device,
        model_id=args.hf_model,
        prompt=args.prompt,
        dtype=args.dtype,
        rtol=args.rtol,
        atol=args.atol,
        max_new_tokens=args.max_new_tokens,
        cpu_only=args.cpu_only,
    )