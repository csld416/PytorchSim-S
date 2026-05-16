import os
import sys
import argparse
import copy
import torch
from transformers import GPT2Config, GPT2LMHeadModel


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


@torch.no_grad()
def run_gpt2_test(
    device,
    batch=1,
    seq_len=32,
    dtype="float32",
    rtol=1e-3,
    atol=1e-3,
    pretrained_name="gpt2",
    do_generate=False,
    max_new_tokens=16,
    cpu_only=False,
):
    print("\n[Running GPT-2 Test]")
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(dtype, torch.float32)

    print(f"Loading GPT-2 model '{pretrained_name}'")
    base_model = GPT2LMHeadModel.from_pretrained(pretrained_name).eval()
    cpu_model = copy.deepcopy(base_model).eval()

    cpu_model.to(dtype=torch_dtype, device="cpu")
    model = base_model.to(dtype=torch_dtype, device=device)

    g = torch.Generator().manual_seed(0)
    vocab = base_model.config.vocab_size
    input_ids_cpu = torch.randint(low=0, high=vocab, size=(batch, seq_len), generator=g, dtype=torch.long)
    attention_mask_cpu = torch.ones((batch, seq_len), dtype=torch.long)

    input_ids_dev = input_ids_cpu.to(device)
    attention_mask_dev = attention_mask_cpu.to(device)

    if cpu_only:
        out_cpu = cpu_model(input_ids=input_ids_cpu, attention_mask=attention_mask_cpu)
        logits_cpu = out_cpu.logits
        print(f"CPU-only logits shape: {tuple(logits_cpu.shape)}")
        if do_generate:
            print("Running CPU-only greedy generate")
            gen_cpu = cpu_model.generate(
                input_ids=input_ids_cpu,
                attention_mask=attention_mask_cpu,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            print(f"CPU-only generated tokens shape: {tuple(gen_cpu.shape)}")
        return

    print("Compiling GPT-2 with torch.compile(...)")
    compiled = torch.compile(model, dynamic=False)

    out_cpu = cpu_model(input_ids=input_ids_cpu, attention_mask=attention_mask_cpu)
    out_dev = compiled(input_ids=input_ids_dev, attention_mask=attention_mask_dev)

    logits_cpu = out_cpu.logits
    logits_dev = out_dev.logits

    test_result("GPT-2 forward(logits)", logits_dev, logits_cpu, rtol=rtol, atol=atol)
    diff = (logits_dev.detach().cpu() - logits_cpu.detach().cpu()).abs().max().item()
    print(f"Max diff > {diff}")

    if do_generate:
        print("Running greedy generate comparison")
        gen_cpu = cpu_model.generate(
            input_ids=input_ids_cpu,
            attention_mask=attention_mask_cpu,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        gen_dev = compiled.generate(
            input_ids=input_ids_dev,
            attention_mask=attention_mask_dev,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        test_result("GPT-2 generate(tokens)", gen_dev, gen_cpu, rtol=0, atol=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test GPT-2 vs CPU (real pretrained weights)")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--pretrained", type=str, default="gpt2")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--cpu_only", action="store_true")
    args = parser.parse_args()

    sys.path.append(os.environ.get("PYTORCHSIM_ROOT_PATH", "/workspace/PyTorchSim"))
    device = torch.device("cpu") if args.cpu_only else torch.device("npu:0")
    if not args.cpu_only:
        torch.compiler.is_compiling = lambda: True

    run_gpt2_test(
        device=device,
        batch=args.batch,
        seq_len=args.seq_len,
        dtype=args.dtype,
        rtol=args.rtol,
        atol=args.atol,
        pretrained_name=args.pretrained,
        do_generate=args.generate,
        max_new_tokens=args.max_new_tokens,
        cpu_only=args.cpu_only,
    )
