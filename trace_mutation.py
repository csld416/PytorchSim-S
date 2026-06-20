import torch._inductor.ir as ir

orig = ir.MutationLayoutSHOULDREMOVE.realize_into

def traced(val, changed_data, unsafe_alias=False):
    print("SLOW PATH HIT, changed_data =", changed_data, type(changed_data))
    return orig(val, changed_data, unsafe_alias)

ir.MutationLayoutSHOULDREMOVE.realize_into = staticmethod(traced)

# now import/run whatever drives the Llama decoder layer compile, e.g.:
# import tests.Llama.test_llama