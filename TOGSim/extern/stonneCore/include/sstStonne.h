#ifndef _SST_STONNE_H
#define _SST_STONNE_H

//sst_stonne.h
#include "STONNEModel.h"
#include "types.h"
#include "SimpleMem.h"

#include "lsQueue.h"


namespace SST_STONNE {
class StonneOpDesc {
public:
    //Input parameters
    /***************************************************************************/
    /*Convolution parameters (See MAERI paper to find out the taxonomy meaning)*/
    /***************************************************************************/
    Layer_t operation;
    std::string layer_name;
    std::string mem_init;
    unsigned int R;                                  // R
    unsigned int S;                                  // S
    unsigned int C;                                  // C
    unsigned int K;                                  // K
    unsigned int G;                                  // G
    unsigned int N;                                  // N
    unsigned int X;                                  // X //TODO CHECK X=1 and Y=1
    unsigned int Y;                                  // Y
    unsigned int X_;                                 // X_
    unsigned int Y_;                                 // Y_
    unsigned int strides;                            // Strides

    //Convolution Tile parameters (See MAERI paper to find out the taxonomy meaning)
    unsigned int T_R;                                // T_R
    unsigned int T_S;                                // T_S
    unsigned int T_C;                                // T_C
    unsigned int T_K;                                // T_K
    unsigned int T_G;                                // T_G
    unsigned int T_N;                                // T_N
    unsigned int T_X_;                               // T_X
    unsigned int T_Y_;                               // T_Y   

    /******************************************************************************/
    /*  GEMM Parameters */
    /******************************************************************************/
    //Layer parameters
    unsigned int GEMM_K;
    unsigned int GEMM_N;
    unsigned int GEMM_M;

    //Tile parameters
    unsigned int GEMM_T_K;
    unsigned int GEMM_T_N;
    unsigned int GEMM_T_M;

    uint64_t matrix_a_dram_address;
    uint64_t matrix_b_dram_address;
    uint64_t matrix_c_dram_address;
    std::string mem_matrix_c_file_name = "";
    std::string bitmap_matrix_a_init = "";
    std::string bitmap_matrix_b_init = "";
    std::string rowpointer_matrix_a_init = "";
    std::string colpointer_matrix_a_init = "";
    std::string rowpointer_matrix_b_init = "";
    std::string colpointer_matrix_b_init = "";
    std::string trace_path = "";
};

class sstStonne {
public:
    sstStonne(std::string config_file, std::string mem_file);
    sstStonne(std::string config_file);
    ~sstStonne();

    // Override SST::Component Virtual Methods
    void setup(StonneOpDesc opDesc, uint64_t offse);
    void finish();
    bool isFinished() { return stonne_instance->isExecutionFinished(); }
    void init( uint32_t phase );
    void pushResponse( SimpleMem::Request* ev );
    SimpleMem::Request* popRequest() { return mem_interface_->popRequest(); }
    //void Status();
    void cycle();
    void printStats() { stonne_instance->printStats(); }
    void printEnergy() { stonne_instance->printEnergy(); }
    LSQueue* loadQueue() { return load_queue_; }
    LSQueue* writeQueue() { return write_queue_; }
    Config getStonneConfig() { return stonne_cfg; }
    MSwitchStats getMSStats() { return stonne_instance->getMSstat(); }
    int getMCFSMStats() { return stonne_instance->getMCFSMStatus(); }
private:
    //SST Variables
    SimpleMem*  mem_interface_ = NULL;
    Stonne* stonne_instance = NULL;

    unsigned int matrixA_size;
    unsigned int matrixB_size;
    unsigned int matrixC_size;

    /**************************************************************************/
    /* Hardware parameters */
    /**************************************************************************/
    Config stonne_cfg; 
    StonneOpDesc opDesc;
    std::string memFileName;
    uint64_t dram_matrixA_address;
    uint64_t dram_matrixB_address;
    uint64_t dram_matrixC_address;

    std::string memMatrixCFileName;

    std::string bitmapMatrixAFileName;
    std::string bitmapMatrixBFileName;

    std::string rowpointerMatrixAFileName;
    std::string colpointerMatrixAFileName;
   
    std::string rowpointerMatrixBFileName;
    std::string colpointerMatrixBFileName;


    /**************************************************************************/
    /* Data pointers */
    /**************************************************************************/

    float* matrixA=NULL;  //This is input ifmaps in CONV operation or MK matrix in GEMM operation
    float* matrixB=NULL;  //This is filter matrix in CONV operation or KN matrix in GEMM operation
    float* matrixC=NULL;  //This is output fmap in CONV operation or resulting MN matrix in GEMM operation

    //These three structures are used to represent the bitmaps in a bitmapSpMSpM operation. The datatype
    //could be smaller to unsigned int (one single bit per element is necessary) but will use this for simplicity 
    //In terms of functionallity the simulation is not affected
    unsigned int* bitmapMatrixA=NULL; //This is the bitmap of MK matrix in bitmapSpMSpM operation
    unsigned int* bitmapMatrixB=NULL; //This is the bitmap of KN matrix in bitmapSpMSpM operation
    unsigned int* bitmapMatrixC=NULL; //This is the bitmap for the resulting matrix in bitmapSpMSpM operation

    unsigned int* rowpointerMatrixA=NULL; //This is the row pointer of MK matrix in csrSpMM operation
    unsigned int* colpointerMatrixA=NULL; //This is the col pointer of MK matrix in csrSpMM operation

    unsigned int* rowpointerMatrixB=NULL;  //This is the pointer of KN matrix in outerProduct operation
    unsigned int* colpointerMatrixB=NULL;  //This is the id pointer of KN matrix in outerProduct operation
    
    /**************************************************************************/
    /* Auxiliary variables */
    /**************************************************************************/
    float EPSILON=0.05;
    unsigned int MAX_RANDOM=10; //Variable used to generate the random values

    /**************************************************************************/
    /* Memory Hierarchy structures and variables */
    /**************************************************************************/

    LSQueue* load_queue_;    //This FIFO stores the load requests sent to the memory hirarchy component
    LSQueue* write_queue_;   //This FIFO stores the write requests sent to the memory hierarchy component

    /**************************************************************************/
    /* Private functions */
    /**************************************************************************/
    std::vector< uint32_t >* constructMemory(std::string fileName);
    unsigned int constructBitmap(std::string fileName, unsigned int * array, unsigned int size);
    void dumpMemoryToFile(std::string fileName, float* array, unsigned int size);
    unsigned int constructCSRStructure(std::string fileName, unsigned int * array);
};
}

#endif
