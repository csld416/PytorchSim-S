#include <iostream>
#include <string>
#include "sstStonne.h"

using namespace SST_STONNE;

int main() {
    StonneOpDesc opDesc;
    opDesc.operation = Layer_t::outerProductGEMM;
    opDesc.GEMM_K = 512;
    opDesc.GEMM_N = 64;
    opDesc.GEMM_M = 64;
    opDesc.GEMM_T_K = 4;
    opDesc.GEMM_T_N = 1;
    opDesc.mem_init = "/workspace/PyTorchSim/PyTorchSimBackend/extern/stonneCore/tests/outerproduct/outerproduct_gemm_mem.ini";
    opDesc.mem_matrix_c_file_name = "/workspace/PyTorchSim/PyTorchSimBackend/extern/stonneCore/tests/outerproduct/result.out";
    opDesc.matrix_a_dram_address = 0;
    opDesc.matrix_b_dram_address = 12444;
    opDesc.matrix_c_dram_address = 24608;
    opDesc.rowpointer_matrix_a_init = "/workspace/PyTorchSim/PyTorchSimBackend/extern/stonneCore/tests/outerproduct/outerproduct_gemm_rowpointerA.in";
    opDesc.colpointer_matrix_a_init = "/workspace/PyTorchSim/PyTorchSimBackend/extern/stonneCore/tests/outerproduct/outerproduct_gemm_colpointerA.in";
    opDesc.rowpointer_matrix_b_init = "/workspace/PyTorchSim/PyTorchSimBackend/extern/stonneCore/tests/outerproduct/outerproduct_gemm_rowpointerB.in";
    opDesc.colpointer_matrix_b_init = "/workspace/PyTorchSim/PyTorchSimBackend/extern/stonneCore/tests/outerproduct/outerproduct_gemm_colpointerB.in";

    std::string hardware_configuration = "/workspace/PyTorchSim/PyTorchSimBackend/extern/stonneCore/tests/sparseflex_op_128mses_128_bw.cfg";

    sstStonne my_stonne(hardware_configuration);
    my_stonne.setup(opDesc);
    my_stonne.init(1);

    while (!my_stonne.isFinished()) {
        my_stonne.cycle();
        if(SimpleMem::Request* req = my_stonne.popRequest()) {
            req->setReply(); 
            my_stonne.pushResponse(req);
        }
    }

    my_stonne.printStats();

    return 0;
}
