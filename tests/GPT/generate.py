"""
CPU-only token generation using the real GPT-NeoX-20B pretrained weights
loaded from HuggingFace transformers.

Usage:
    python generate.py [--model_id <hf_model_id>] [--prompt <text>]
                       [--max_new_tokens <n>] [--temperature <t>]
                       [--top_k <k>] [--top_p <p>]
                       [--stream_layers]
"""

import argparse
import json
import os
import time
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from accelerate.utils import set_module_tensor_to_device
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)


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


class StreamedCheckpointLoader:
    """Reads weights straight out of the HF safetensors shards, one submodule at a time, so the
    full ~40GB GPT-NeoX-20B checkpoint never has to be materialized in memory at once."""

    def __init__(self, model_id_or_path):
        self.snapshot_dir = (
            model_id_or_path
            if os.path.isdir(model_id_or_path)
            else snapshot_download(model_id_or_path, allow_patterns=["*.safetensors", "*.safetensors.index.json"])
        )
        index_path = os.path.join(self.snapshot_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                self.weight_map = json.load(f)["weight_map"]
        else:
            shard_name = "model.safetensors"
            with safe_open(os.path.join(self.snapshot_dir, shard_name), framework="pt") as f:
                self.weight_map = {name: shard_name for name in f.keys()}
        self._handles = {}

    def _handle(self, filename):
        handle = self._handles.get(filename)
        if handle is None:
            handle = safe_open(os.path.join(self.snapshot_dir, filename), framework="pt")
            self._handles[filename] = handle
        return handle

    def load_module_(self, module, prefix, device, dtype):
        """Materializes `module`'s parameters from the checkpoint onto `device`."""
        for name, _ in list(module.named_parameters(recurse=True)):
            full_name = f"{prefix}.{name}"
            tensor = self._handle(self.weight_map[full_name]).get_tensor(full_name).to(dtype=dtype)
            set_module_tensor_to_device(module, name, device, value=tensor)

    def unload_module_(self, module):
        """Frees `module`'s parameters back to the meta device (no storage retained)."""
        for name, _ in list(module.named_parameters(recurse=True)):
            set_module_tensor_to_device(module, name, "meta")


def _build_meta_model(config, torch_dtype):
    """Builds the GPT-NeoX module tree on the meta device (no weight memory used yet). Rotary
    embeddings and the causal-mask bias are non-persistent buffers derived from config rather than
    the checkpoint, so they come out as meta tensors too under the meta context; re-run each
    attention module's own `_init_rope`/`_init_bias` to give them real values."""
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch_dtype)
    model.eval()

    for layer in model.gpt_neox.layers:
        attn = layer.attention
        attn._init_bias(config.max_position_embeddings)
        attn._init_rope()
    return model


@torch.no_grad()
def _forward_streamed(model, loader, input_ids, attention_mask, past_key_values, device, dtype):
    """Equivalent to GPTNeoXForCausalLM.forward(), except each decoder layer's weights are loaded
    from disk right before it runs and discarded right after, so only one layer's worth of weights
    (~450MB for GPT-NeoX-20B) is resident at a time instead of the full ~40GB model."""
    base_model = model.gpt_neox
    batch_size, seq_length = input_ids.shape

    if past_key_values is None:
        past_length = 0
        past_key_values = tuple([None] * base_model.config.num_hidden_layers)
    else:
        past_length = past_key_values[0][0].size(-2)

    position_ids = torch.arange(
        past_length, seq_length + past_length, dtype=torch.long, device=input_ids.device
    ).unsqueeze(0)

    inputs_embeds = base_model.embed_in(input_ids)

    flat_attention_mask = attention_mask.view(batch_size, -1) if attention_mask is not None else None
    if base_model._attn_implementation == "sdpa":
        causal_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask=flat_attention_mask,
            input_shape=(batch_size, seq_length),
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_length,
        )
    else:
        causal_mask = _prepare_4d_causal_attention_mask(
            attention_mask=flat_attention_mask,
            input_shape=(batch_size, seq_length),
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_length,
        )

    hidden_states = inputs_embeds
    presents = []
    for layer_idx, layer in enumerate(base_model.layers):
        loader.load_module_(layer, f"gpt_neox.layers.{layer_idx}", device, dtype)
        layer_outputs = layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            layer_past=past_key_values[layer_idx],
            use_cache=True,
        )
        hidden_states = layer_outputs[0]
        presents.append(layer_outputs[1])
        loader.unload_module_(layer)

    hidden_states = base_model.final_layer_norm(hidden_states)
    logits = model.embed_out(hidden_states).float()
    return logits, tuple(presents)


def _sample_next_token(logits, temperature, top_k, top_p):
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth_vals = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = torch.where(logits < kth_vals, torch.full_like(logits, float("-inf")), logits)
    if top_p > 0.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate_streamed(
    prompt: str,
    model_id: str = DEFAULT_MODEL_ID,
    max_new_tokens: int = 3,
    temperature: float = 0.0,
    top_k: int = 50,
    top_p: float = 0.9,
    dtype: str = "bfloat16",
):
    torch.manual_seed(0)
    torch_dtype = DTYPE_MAP[dtype]
    device = torch.device("cpu")
    print(f"Streaming generation with model {model_id} (dtype={torch_dtype}, layer-by-layer on CPU) ...")

    print(f"Loading tokenizer/config from {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    config = AutoConfig.from_pretrained(model_id)

    print("Building model skeleton on the meta device (no weights loaded yet)")
    model = _build_meta_model(config, torch_dtype)

    print("Resolving local checkpoint shards")
    loader = StreamedCheckpointLoader(model_id)

    print("Loading persistent weights (embed_in, final_layer_norm, embed_out)")
    loader.load_module_(model.gpt_neox.embed_in, "gpt_neox.embed_in", device, torch_dtype)
    loader.load_module_(model.gpt_neox.final_layer_norm, "gpt_neox.final_layer_norm", device, torch_dtype)
    loader.load_module_(model.embed_out, "embed_out", device, torch_dtype)

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    print(f"\nPrompt ({input_ids.shape[1]} tokens): {prompt!r}")

    gen_ids = input_ids
    gen_mask = torch.ones_like(gen_ids)
    past_key_values = None
    print("Generating (streaming one decoder layer at a time)...")
    for step in range(max_new_tokens):
        step_input_ids = gen_ids if step == 0 else gen_ids[:, -1:]
        logits, past_key_values = _forward_streamed(
            model, loader, step_input_ids, gen_mask, past_key_values, device, torch_dtype
        )
        next_token = _sample_next_token(logits[:, -1, :], temperature, top_k, top_p)
        gen_ids = torch.cat([gen_ids, next_token], dim=1)
        gen_mask = torch.cat([gen_mask, torch.ones_like(next_token)], dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            print("EOS reached, stopping early.")
            break

        generated_text = tokenizer.decode(gen_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
        print(f"\nOutput:{generated_text}")
        


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
    parser.add_argument("--stream_layers", action="store_true",
                         help="Load one decoder layer's weights at a time instead of the whole model (CPU only)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.stream_layers:
        generate_streamed(
            prompt=args.prompt,
            model_id=args.model_id,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            dtype=args.dtype,
        )
    else:
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
