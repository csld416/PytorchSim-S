import torch

from PyTorchSimFrontend.weight_placement import write_weight_placements

def test_result(name, out, cpu_out, rtol=1e-4, atol=1e-4):
    if torch.allclose(out.cpu(), cpu_out, rtol=rtol, atol=atol):
        message = f"|{name} Test Passed|"
        print("-" * len(message))
        print(message)
        print("-" * len(message))
    else:
        message = f"|{name} Test Failed|"
        print("-" * len(message))
        print(message)
        print("-" * len(message))
        print("custom out: ", out.cpu())
        print("cpu out: ", cpu_out)
        exit(1)

def test_matmul(device, input_size=128, hidden_size=128, output_size=128):
    def custom_matmul(a, b):
        return torch.matmul(a, b)
    torch.manual_seed(0)
    input = torch.randn(input_size, hidden_size, dtype=torch.float16)
    # Place the actual matmul weight after a prefix in the same allocation.
    # Registering the full backing tensor but passing this contiguous view to
    # the NPU forces DMA address translation to preserve a real, nonzero
    # within-tensor displacement instead of requesting the placement base.
    weight_prefix_rows = 4
    weight_backing = torch.randn(
        hidden_size + weight_prefix_rows, output_size, dtype=torch.float16
    )
    weight = weight_backing[weight_prefix_rows:]
    x1 = input.to(device=device)
    weight_backing_npu = weight_backing.to(device=device)
    w1 = weight_backing_npu[weight_prefix_rows:]
    expected_weight_displacement = (
        weight_prefix_rows * output_size * weight.element_size()
    )
    actual_weight_displacement = w1.data_ptr() - weight_backing_npu.data_ptr()
    assert actual_weight_displacement == expected_weight_displacement
    # Keep a one-page validation placement ahead of the real weight so the
    # end-to-end SSD test proves a nonzero logical-address translation rather
    # than being indistinguishable from the old hard-coded offset-zero path.
    placement_sentinel = torch.empty(2048, dtype=torch.float16).to(device=device)
    write_weight_placements((
        ("tests.matmul.validation_sentinel", placement_sentinel),
        ("tests.matmul.weight_backing", weight_backing_npu),
    ))
    print(
        "NUSSD subrange proof: "
        f"weight_host_base={weight_backing_npu.data_ptr()} "
        f"dma_host_base={w1.data_ptr()} "
        f"displacement_bytes={actual_weight_displacement}"
    )
    x2 = input.to("cpu")
    w2 = weight.to("cpu")
    opt_fn = torch.compile(dynamic=False)(custom_matmul)
    res = opt_fn(x1, w1)
    y = custom_matmul(x2, w2)
    # TOGSim's FP16 accumulation differs slightly from the CPU implementation.
    # An SSD-disabled 1024^3 baseline measured max_abs=0.15625 and
    # mean_abs=0.0107502546; these bounds pass that established compute path
    # while remaining tight enough to catch material corruption.
    test_result("Matmul Forward", res, y, rtol=1e-2, atol=1e-1)

def test_addmm(device, input_size=128, hidden_size=128, output_size=128, bias_rank=1):
    def custom_matmul(bias, a, b):
        return torch.addmm(bias, a, b)
    torch.manual_seed(0)
    input = torch.randn(input_size, hidden_size, dtype=torch.float16)
    weight = torch.randn(hidden_size, output_size, dtype=torch.float16)
    bias = torch.randn(output_size, dtype=torch.float16) if bias_rank == 1 else torch.randn(input_size, output_size, dtype=torch.float16)
    x1 = input.to(device=device)
    w1 = weight.to(device=device)
    b1 = bias.to(device=device)
    x2 = input.to("cpu")
    w2 = weight.to("cpu")
    b2 = bias.to("cpu")
    opt_fn = torch.compile(dynamic=False)(custom_matmul)
    res = opt_fn(b1, x1, w1)
    y = custom_matmul(b2, x2, w2)
    test_result("Addmm Forward", res, y)

def test_addmm2(device, input_size=128, hidden_size=128, output_size=128):
    def custom_matmul(bias, a, b):
        return torch.matmul(a, b) #+ bias
    torch.manual_seed(0)
    input = torch.randn(input_size, hidden_size, dtype=torch.float16)
    weight = torch.randn(hidden_size, output_size, dtype=torch.float16)
    bias = torch.randn(input_size, 1, dtype=torch.float16)
    x1 = input.to(device=device)
    w1 = weight.to(device=device)
    b1 = bias.to(device=device)
    x2 = input.to("cpu")
    w2 = weight.to("cpu")
    b2 = bias.to("cpu")
    opt_fn = torch.compile(dynamic=False)(custom_matmul)
    res = opt_fn(b1, x1, w1)
    y = custom_matmul(b2, x2, w2)
    test_result("Addmm2 Forward", res, y)

def test_linear(device, input_size=128, hidden_size=128, output_size=128):
    def custom_linear(a, b, bias):
        linear = torch.nn.Linear(hidden_size, output_size)
        linear.weight = torch.nn.Parameter(b)
        linear.bias = torch.nn.Parameter(bias)
        return linear(a)
    torch.manual_seed(0)
    input = torch.randn(input_size, hidden_size, dtype=torch.float16)
    weight = torch.randn(output_size, hidden_size, dtype=torch.float16)
    bias = torch.randn(output_size, dtype=torch.float16)
    x1 = input.to(device=device)
    w1 = weight.to(device=device)
    b1 = bias.to(device=device)
    x2 = input.to("cpu")
    w2 = weight.to("cpu")
    b2 = bias.to("cpu")
    opt_fn = torch.compile(dynamic=False)(custom_linear)
    res = opt_fn(x1, w1, b1)
    y = custom_linear(x2, w2, b2)
    test_result("Linear Forward", res, y)

if __name__ == "__main__":
    device = torch.device("npu:0")
    test_matmul(device, 1024, 1024, 1024)
    # test_matmul(device, 32, 32, 32)
    # test_matmul(device, 128, 128, 128)
    # test_matmul(device, 256, 256, 256)
    # test_matmul(device, 128, 256, 256)
    # test_matmul(device, 128, 63, 56)
    # test_addmm(device, 128, 256, 512)
    # test_addmm(device, 128, 256, 512, bias_rank=2)
    # test_addmm(device, 129, 61, 56)
    # test_addmm2(device, 129, 61, 56)
    # test_addmm(device, 129*4, 61*4, 56*4)
    # test_addmm2(device, 129*4, 61*4, 56*4)
