import os
import sys
import argparse
import copy
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_result(name, out, ref, rtol=1e-4, atol=1e-4):
    if torch.allclose(out.cpu(), ref.cpu(), rtol=rtol, atol=atol):
        msg = f"|{name} Test Passed|"
        print("-" * len(msg))
        print(msg)
        print("-" * len(msg))
    else:
        msg = f"|{name} Test Failed|"
        print("-" * len(msg))
        print(msg)
        print("-" * len(msg))
        diff = (out.cpu().float() - ref.cpu().float()).abs().max().item()
        print("device out:", out.detach().cpu())
        print("cpu ref  :", ref.detach().cpu())
        print(f"Max abs diff: {diff}")
        sys.exit(1)


def dump_fx_graph(model, example_inputs, output_path):
    gm, _ = torch._dynamo.export(model, *example_inputs)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(str(gm.graph))
        f.write("\n\n")
        f.write(gm.print_readable())


@torch.no_grad()
def run_opt_350m_test(
    device,
    prompt="Hello!",
    dtype="float32",
    rtol=1e-3,
    atol=1e-3,
    model_id="facebook/opt-350m",
    do_generate=False,
    max_new_tokens=16,
    cpu_only=False,
    dump_fx_path=None,
):
    print("\n[Running OPT-350M Test]")
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(dtype, torch.float32)

    print(f"Loading OPT model '{model_id}'")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=None,
    ).eval()
    cpu_model = copy.deepcopy(base_model).eval()

    cpu_model.to(dtype=torch_dtype, device="cpu")
    dev_model = base_model.to(dtype=torch_dtype, device=device)

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids_cpu = inputs["input_ids"]
    attention_mask_cpu = inputs["attention_mask"]

    input_ids_dev = input_ids_cpu.to(device)
    attention_mask_dev = attention_mask_cpu.to(device)

    if dump_fx_path:
        print(f"Dumping FX graph to {dump_fx_path}")
        dump_fx_graph(dev_model, (input_ids_dev, attention_mask_dev), dump_fx_path)

    if cpu_only:
        if do_generate:
            print("Running CPU-only greedy generate")
            gen_cpu = cpu_model.generate(
                input_ids=input_ids_cpu,
                attention_mask=attention_mask_cpu,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            new_ids = gen_cpu[:, input_ids_cpu.shape[1]:]
            new_text = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
            print(f"CPU-only generated tokens shape: {tuple(gen_cpu.shape)}")
            print(f"CPU-only generated text: {new_text}")
        else:
            print("Running CPU-only forward...")
            out_cpu = cpu_model(input_ids=input_ids_cpu, attention_mask=attention_mask_cpu)
            logits_cpu = out_cpu.logits
            print(f"CPU-only logits shape: {tuple(logits_cpu.shape)}")
        return

    print("Compiling OPT-350M with torch.compile(...)")
    compiled = torch.compile(dev_model, dynamic=False)

    out_cpu = cpu_model(input_ids=input_ids_cpu, attention_mask=attention_mask_cpu)
    out_dev = compiled(input_ids=input_ids_dev, attention_mask=attention_mask_dev)

    logits_cpu = out_cpu.logits[:, -1, :]
    logits_dev = out_dev.logits[:, -1, :]

    # test_result("OPT-350M logits (last token)", logits_dev, logits_cpu, rtol=rtol, atol=atol)
    diff = (logits_dev.detach().cpu() - logits_cpu.detach().cpu()).abs().max().item()
    print(f"Max diff > {diff}")

    # if do_generate:
    #     print("Running greedy generate comparison")
    #     gen_cpu = cpu_model.generate(
    #         input_ids=input_ids_cpu,
    #         attention_mask=attention_mask_cpu,
    #         max_new_tokens=max_new_tokens,
    #         do_sample=False,
    #     )
    #     gen_dev = compiled.generate(
    #         input_ids=input_ids_dev,
    #         attention_mask=attention_mask_dev,
    #         max_new_tokens=max_new_tokens,
    #         do_sample=False,
    #     )
    #     test_result("OPT-350M generate(tokens)", gen_dev, gen_cpu, rtol=0, atol=0)


@torch.no_grad()
def run_opt_350m_layers(
    device,
    prompt="Hello!",
    dtype="float32",
    rtol=1e-3,
    atol=1e-3,
    model_id="facebook/opt-350m",
    max_layers=None,
    compile_per_prefix=False,
    cpu_only=False,
    dump_fx_path=None,
):
    print("\n[Running OPT-350M Blockwise Test]")
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(dtype, torch.float32)

    print(f"Loading OPT model '{model_id}'")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=None,
    ).eval()
    cpu_model = copy.deepcopy(base_model).eval()

    cpu_model.to(dtype=torch_dtype, device="cpu")
    dev_model = base_model.to(dtype=torch_dtype, device=device)

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids_cpu = inputs["input_ids"]
    attention_mask_cpu = inputs["attention_mask"]

    input_ids_dev = input_ids_cpu.to(device)
    attention_mask_dev = attention_mask_cpu.to(device)

    if dump_fx_path:
        print(f"Dumping FX graph to {dump_fx_path}")
        dump_fx_graph(dev_model, (input_ids_dev, attention_mask_dev), dump_fx_path)

    cpu_decoder = cpu_model.model.decoder
    dev_decoder = dev_model.model.decoder
    cpu_layers = list(cpu_decoder.layers)
    dev_layers = list(dev_decoder.layers)

    num_layers = len(dev_layers)
    limit = num_layers if max_layers is None else min(max_layers, num_layers)
    print(f"Total layers: {num_layers}. Running prefix up to {limit} layer(s).")

    for i in range(1, limit + 1):
        cpu_decoder.layers = nn.ModuleList(cpu_layers[:i])
        dev_decoder.layers = nn.ModuleList(dev_layers[:i])

        run_dev_model = dev_model
        if compile_per_prefix and not cpu_only:
            print(f"Compiling OPT-350M prefix {i}/{num_layers} with torch.compile(...)")
            run_dev_model = torch.compile(dev_model, dynamic=False)

        out_cpu = cpu_decoder(
            input_ids=input_ids_cpu,
            attention_mask=attention_mask_cpu,
            use_cache=False,
            output_hidden_states=False,
        )
        hidden_cpu = out_cpu.last_hidden_state

        if cpu_only:
            print(f"[Layers {i}/{num_layers}] CPU hidden: {tuple(hidden_cpu.shape)}")
            continue

        out_dev = run_dev_model.model.decoder(
            input_ids=input_ids_dev,
            attention_mask=attention_mask_dev,
            use_cache=False,
            output_hidden_states=False,
        )
        hidden_dev = out_dev.last_hidden_state

        test_result(
            f"OPT-350M prefix {i} hidden(last token)",
            hidden_dev[:, -1, :],
            hidden_cpu[:, -1, :],
            rtol=rtol,
            atol=atol,
        )
        diff = (hidden_dev[:, -1, :].detach().cpu() - hidden_cpu[:, -1, :].detach().cpu()).abs().max().item()
        print(f"[Layers {i}/{num_layers}] Max diff > {diff}")

    cpu_decoder.layers = nn.ModuleList(cpu_layers)
    dev_decoder.layers = nn.ModuleList(dev_layers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test OPT-350M vs CPU (real pretrained weights)")
    parser.add_argument("--prompt", type=str, default="Hello!")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--model_id", type=str, default="facebook/opt-350m")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--cpu_only", action="store_true")
    parser.add_argument("--blockwise", action="store_true")
    parser.add_argument("--max_layers", type=int, default=None)
    parser.add_argument("--compile_per_prefix", action="store_true")
    parser.add_argument("--dump_fx", type=str, default=None)
    args = parser.parse_args()

    sys.path.append(os.environ.get("PYTORCHSIM_ROOT_PATH", "/workspace/PyTorchSim"))
    device = torch.device("cpu") if args.cpu_only else torch.device("npu:0")
    if not args.cpu_only:
        torch.compiler.is_compiling = lambda: True

    if args.blockwise:
        run_opt_350m_layers(
            device=device,
            prompt=args.prompt,
            dtype=args.dtype,
            rtol=args.rtol,
            atol=args.atol,
            model_id=args.model_id,
            max_layers=args.max_layers,
            compile_per_prefix=args.compile_per_prefix,
            cpu_only=args.cpu_only,
            dump_fx_path=args.dump_fx,
        )
    else:
        run_opt_350m_test(
            device=device,
            prompt=args.prompt,
            dtype=args.dtype,
            rtol=args.rtol,
            atol=args.atol,
            model_id=args.model_id,
            do_generate=args.generate,
            max_new_tokens=args.max_new_tokens,
            cpu_only=args.cpu_only,
            dump_fx_path=args.dump_fx,
        )
