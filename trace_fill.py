import traceback
import torch
from torch.utils._python_dispatch import TorchDispatchMode

class FillTracer(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if "fill_" in str(func):
            print("=== fill_ dispatch:", func, "===")
            traceback.print_stack()
            print("=== end stack ===")
        return func(*args, **(kwargs or {}))

_tracer = FillTracer()
_tracer.__enter__()

# import/run the llama decoder test below this point, e.g.:
# import tests.Llama.test_llama