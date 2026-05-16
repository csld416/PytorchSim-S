//sstStonne.cc
#include "sstStonne.h"
#include <math.h>
#include <iostream>
#include <iomanip>
#include "utility.h"
using namespace SST_STONNE;

//Constructor
sstStonne::sstStonne(std::string config_file, std::string mem_file) :
    stonne_cfg(Config(config_file)), memFileName(mem_file) {
    //set up memory interfaces
    mem_interface_ = new SimpleMem();

    write_queue_ = new LSQueue();
    load_queue_ = new LSQueue();
    stonne_instance = new Stonne(stonne_cfg, load_queue_, write_queue_, mem_interface_);
}

sstStonne::sstStonne(std::string config_file) : sstStonne(config_file, "") {};

sstStonne::~sstStonne() {}

void sstStonne::init( uint32_t phase )
{
    mem_interface_->init( phase );

    if( 0 == phase ) {
        std::vector< uint32_t >* initVector;

        //Check to see if there is any memory being initialized
        if( memFileName != "" ) {
            initVector = constructMemory(memFileName);
        } else {
            initVector = new std::vector< uint32_t > {16, 64, 32, 0 , 16382, 0, 0};
        }

        std::vector<uint8_t> memInit;
        constexpr auto buff_size = sizeof(uint32_t);
        uint8_t buffer[buff_size] = {};
        for( auto it = initVector->begin(); it != initVector->end(); ++it ) {
            std::memcpy(buffer, std::addressof(*it), buff_size);
            for( uint32_t i = 0; i < buff_size; ++i ){
                memInit.push_back(buffer[i]);
            }
        }

        SimpleMem::Request* initMemory = new SimpleMem::Request(SimpleMem::Request::Write, 0, memInit.size(), memInit);
        mem_interface_->sendInitData(initMemory);
    }
}


void sstStonne::cycle() {
    stonne_instance->cycle();
}

void sstStonne::setup(StonneOpDesc operation, uint64_t offset=0) {
    //Creating arrays for this version of the integration
    opDesc = operation;
    dram_matrixA_address = operation.matrix_a_dram_address + offset;
    dram_matrixB_address = operation.matrix_b_dram_address + offset;
    dram_matrixC_address = operation.matrix_c_dram_address + offset;

    memMatrixCFileName = opDesc.mem_matrix_c_file_name;
    bitmapMatrixAFileName = opDesc.bitmap_matrix_a_init;
    bitmapMatrixBFileName = opDesc.bitmap_matrix_b_init;

    rowpointerMatrixAFileName = opDesc.rowpointer_matrix_a_init;
    colpointerMatrixAFileName = opDesc.colpointer_matrix_a_init;
    rowpointerMatrixBFileName = opDesc.rowpointer_matrix_b_init;
    colpointerMatrixBFileName = opDesc.colpointer_matrix_b_init;

    if((opDesc.operation==CONV) || (opDesc.operation==GEMM)) { //Initializing dense operation
        switch(opDesc.operation) {
            case CONV:
                matrixA_size=opDesc.N*opDesc.X*opDesc.Y*opDesc.C; //ifmap
                matrixB_size=opDesc.R*opDesc.S*(opDesc.C/opDesc.G)*opDesc.K;
                matrixC_size=opDesc.N*opDesc.X_*opDesc.Y_*opDesc.K;
                break;
            case GEMM:
                matrixA_size=opDesc.GEMM_M*opDesc.GEMM_K;
                matrixB_size=opDesc.GEMM_N*opDesc.GEMM_K;
                matrixC_size=opDesc.GEMM_M*opDesc.GEMM_N;
                break;
            default:
                break;
        }
        matrixA = NULL; //TODO: remove when everything is done
        matrixB = NULL;
        matrixC = new float[matrixC_size];
    } //End initializing dense operation

    else { //Initializing sparse operation
        if(opDesc.operation==bitmapSpMSpM) {
            if(bitmapMatrixAFileName=="") {

            }
            if(bitmapMatrixBFileName=="") {

            }

            matrixA_size=opDesc.GEMM_M*opDesc.GEMM_K;
            matrixB_size=opDesc.GEMM_N*opDesc.GEMM_K;
            matrixC_size=opDesc.GEMM_M*opDesc.GEMM_N;
            bitmapMatrixA=new unsigned int[matrixA_size];
            bitmapMatrixB=new unsigned int[matrixB_size];
            bitmapMatrixC=new unsigned int[matrixC_size];
            unsigned int nActiveValuesA=constructBitmap(bitmapMatrixAFileName, bitmapMatrixA, matrixA_size);
            unsigned int nActiveValuesB=constructBitmap(bitmapMatrixBFileName, bitmapMatrixB, matrixB_size);
            matrixC=new float[matrixC_size];
        } //End bitmapSpMSpM operation

        else if(opDesc.operation==csrSpMM) {
            if(rowpointerMatrixAFileName=="") {
            }
            if(colpointerMatrixAFileName=="") {
            }
            matrixA_size=opDesc.GEMM_M*opDesc.GEMM_K;
            matrixB_size=opDesc.GEMM_N*opDesc.GEMM_K;
            matrixC_size=opDesc.GEMM_M*opDesc.GEMM_N;

            rowpointerMatrixA=new unsigned int[matrixA_size]; //Change to the minimum using vector class
            colpointerMatrixA=new unsigned int[matrixA_size];
            unsigned int nValuesRowPointer=constructCSRStructure(rowpointerMatrixAFileName,rowpointerMatrixA);
            unsigned int nValuesColPointer=constructCSRStructure(colpointerMatrixAFileName, colpointerMatrixA);
            matrixA=NULL; //TODO fix this
            matrixB=NULL;
            matrixC=new float[matrixC_size];
            // Data is not mandatory

        } //End csrSpMM operation

        else if((opDesc.operation==outerProductGEMM) || (opDesc.operation==gustavsonsGEMM)) {
            matrixA_size=opDesc.GEMM_M*opDesc.GEMM_K;
            matrixB_size=opDesc.GEMM_N*opDesc.GEMM_K;
            matrixC_size=opDesc.GEMM_M*opDesc.GEMM_N;

            rowpointerMatrixA=new unsigned int[matrixA_size+1]; //Change to the minimum using vector class
            colpointerMatrixA=new unsigned int[matrixA_size+1];

            rowpointerMatrixB = new unsigned int[matrixB_size+1];
            colpointerMatrixB = new unsigned int[matrixB_size+1];

            unsigned int nValuesRowPointerA=constructCSRStructure(rowpointerMatrixAFileName, rowpointerMatrixA);
            unsigned int nValuesColPointerA=constructCSRStructure(colpointerMatrixAFileName, colpointerMatrixA);

            unsigned int nValuesRowPointerB=constructCSRStructure(rowpointerMatrixBFileName, rowpointerMatrixB);
            unsigned int nValuesColPointerB=constructCSRStructure(colpointerMatrixBFileName, colpointerMatrixB);
            matrixA=NULL; //TODO fix this
            matrixB=NULL;
            matrixC=new float[matrixC_size];
            // Data is not mandatory
        }
    }

    stonne_instance->loadAddress(dram_matrixA_address, dram_matrixB_address, dram_matrixC_address);
    switch(opDesc.operation) {
        case CONV:
            stonne_instance->loadDNNLayer(CONV, opDesc.layer_name, opDesc.R, opDesc.S, opDesc.C, opDesc.K,
                opDesc.G, opDesc.N, opDesc.X, opDesc.Y, opDesc.strides, matrixA, matrixB, matrixC, CNN_DATAFLOW); //Loading the layer
            stonne_instance->loadTile(opDesc.T_R, opDesc.T_S, opDesc.T_C, opDesc.T_K, opDesc.T_G, opDesc.T_N, opDesc.T_X_, opDesc.T_Y_);
            break;
        case GEMM:
            stonne_instance->loadDenseGEMM(opDesc.layer_name, opDesc.GEMM_N, opDesc.GEMM_K, opDesc.GEMM_M, matrixA, matrixB, matrixC, CNN_DATAFLOW);
            stonne_instance->loadGEMMTile(opDesc.GEMM_T_N, opDesc.GEMM_T_K, opDesc.GEMM_T_M);
                break;
        case bitmapSpMSpM:
            stonne_instance->loadGEMM(opDesc.layer_name, opDesc.GEMM_N, opDesc.GEMM_K, opDesc.GEMM_M, matrixA, matrixB, bitmapMatrixA, bitmapMatrixB, matrixC, bitmapMatrixC, MK_STA_KN_STR );
            break;
        case csrSpMM:
            stonne_instance->loadSparseDense(opDesc.layer_name, opDesc.GEMM_N, opDesc.GEMM_K, opDesc.GEMM_M, matrixA, matrixB, colpointerMatrixA, rowpointerMatrixA, matrixC, opDesc.GEMM_T_N, opDesc.GEMM_T_K);
            break;
        case outerProductGEMM:
            stonne_instance->loadSparseOuterProduct(opDesc.layer_name, opDesc.GEMM_N, opDesc.GEMM_K, opDesc.GEMM_M, matrixA, matrixB, colpointerMatrixA, rowpointerMatrixA, colpointerMatrixB, rowpointerMatrixB, matrixC);
            break;
        case gustavsonsGEMM:
            stonne_instance->loadSparseOuterProduct(opDesc.layer_name, opDesc.GEMM_N, opDesc.GEMM_K, opDesc.GEMM_M, matrixA, matrixB, colpointerMatrixA, rowpointerMatrixA, colpointerMatrixB, rowpointerMatrixB, matrixC);
            break;
        default:
            break;
    };
    stonne_instance->printStats();
}

std::vector< uint32_t >* sstStonne::constructMemory(std::string fileName) { //In the future version this will be directly simulated memory
    std::vector< uint32_t >* tempVector = new std::vector< uint32_t >;
    std::ifstream inputStream(fileName, std::ios::in);
    if( inputStream.is_open() ) {
        std::string thisLine;
        while( std::getline( inputStream, thisLine ) )
        {
            std::string value;
            std::stringstream stringIn(thisLine);
            while( std::getline(stringIn, value, ',') ) {
                tempVector->push_back(std::stoul(value));
            }
        }
        inputStream.close();
    } else {
        exit(0);
    }
    return tempVector;
}

//Return number of active elements to build later the data array. This is necessary because we do not know the number of elements.
unsigned int sstStonne::constructBitmap(std::string fileName, unsigned int * array, unsigned int size) { //In the future version this will be directly simulated memory
    std::ifstream inputStream(fileName, std::ios::in);
    unsigned int currentIndex=0;
    unsigned int nActiveValues=0;
    if( inputStream.is_open() ) {
        std::string thisLine;
        while( std::getline( inputStream, thisLine ) )
        {
            std::string value;
            std::stringstream stringIn(thisLine);
            while( std::getline(stringIn, value, ',') ) {
                array[currentIndex]=stoi(value);
                currentIndex++;
                if(stoi(value)==1) {
                        nActiveValues++;
                }
            }
        }
        inputStream.close();
    } else {
        exit(0);
    }
    return nActiveValues;
}

//Return number of active elements to build later the data array. This is necessary because we do not know the number of elements.
unsigned int sstStonne::constructCSRStructure(std::string fileName, unsigned int * array) { //In the future version this will be directly simulated memory
    std::ifstream inputStream(fileName, std::ios::in);
    unsigned int currentIndex=0;
    if( inputStream.is_open() ) {
        std::string thisLine;
        while( std::getline( inputStream, thisLine ) )
        {
            std::string value;
            std::stringstream stringIn(thisLine);
            while( std::getline(stringIn, value, ',') ) {
                array[currentIndex]=stoi(value);
                currentIndex++;
            }
        }
        inputStream.close();
    } else {
        exit(0);
    }
    return currentIndex;
}

void sstStonne::finish() {
    //This code should have the logic to write the output memory into a certain file passed by parameter. TODO
    dumpMemoryToFile(memMatrixCFileName, matrixC, matrixC_size);
    //delete stonne_instance;
    //delete[] matrixC;

    if(opDesc.operation==bitmapSpMSpM) {
        if (bitmapMatrixA)
            delete[] bitmapMatrixA;
        if (bitmapMatrixB)
            delete[] bitmapMatrixB;
        bitmapMatrixA = NULL;
        bitmapMatrixB = NULL;
    }

    else if(opDesc.operation==csrSpMM) {
        if (rowpointerMatrixA)
            delete[] rowpointerMatrixA;
        if (colpointerMatrixA)
            delete[] colpointerMatrixA;
        rowpointerMatrixA = NULL;
        colpointerMatrixA = NULL;
    }
    else if((opDesc.operation==outerProductGEMM) || (opDesc.operation==gustavsonsGEMM)) {
        if (rowpointerMatrixA)
            delete[] rowpointerMatrixA;
        if (colpointerMatrixA)
            delete[] colpointerMatrixA;
        if (rowpointerMatrixB)
            delete[] rowpointerMatrixB;
        if (colpointerMatrixB)
            delete[] colpointerMatrixB;
        rowpointerMatrixA = NULL;
        colpointerMatrixA = NULL;
        rowpointerMatrixB = NULL;
        colpointerMatrixB = NULL;
    }
    //std::cout << "The finish function ends" << std::endl;
}

void sstStonne::dumpMemoryToFile(std::string fileName, float* array, unsigned int size) {
    if(fileName != "") {
        std::ofstream outputStream (fileName, std::ios::out);
        if( outputStream.is_open()) {
            for(unsigned i=0; i<size; i++) {
                float value = array[i];
                outputStream << std::fixed << std::setprecision(1) << value << ",";
            }
            outputStream.close();
        }
        else {
            exit(0);
        }
    }
}

void sstStonne::pushResponse( SimpleMem::Request* ev ) {
    if( ev->cmd == SimpleMem::Request::Command::ReadResp ) {
        // Read request needs some special handling
        data_t memValue = 0.0;

        std::memcpy( std::addressof(memValue), std::addressof(ev->data[0]), sizeof(memValue) );
        load_queue_->setEntryData( ev->id, memValue);
        load_queue_->setEntryReady( ev->id, 1 );
    } else if ( ev->cmd == SimpleMem::Request::Command::WriteResp) {
        //output_->verbose(CALL_INFO, 8, 0, "Response to a write for addr: %" PRIu64 " to PE %" PRIu32 "\n",
        //                 ev->addr, ls_queue_->lookupEntry( ev->id ).second );
        write_queue_->setEntryReady( ev->id, 1 );
    } else {
        //error log...
    }

    // Need to clean up the events coming back from the cache
    delete ev;
}
