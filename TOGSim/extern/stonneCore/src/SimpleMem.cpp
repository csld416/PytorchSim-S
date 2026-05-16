#include "SimpleMem.h"

uint32_t SimpleMem::Request::id_seed=0;

void SimpleMem::Request::setReply() {
    switch (cmd)
    {
    case Command::Read:
        cmd = Command::ReadResp;
        break;
    case Command::Write:
        cmd = Command::WriteResp;
        break;
    default:
        std::cout <<"[setReply] Unexpected request type\n";
        exit(1);
    }
}
// Constructor
SimpleMem::SimpleMem() {
}

void SimpleMem::init(int phase) {
}

void SimpleMem::sendInitData(Request* req) {
    if (!req) {
        std::cerr << "[Error] sendInitData called with null request!" << std::endl;
        return;
    }
    sendRequest(req);
}

void SimpleMem::sendRequest(Request* req) {
    if (!req) {
        std::cerr << "[Error] processRequest called with null request!" << std::endl;
        return;
    }

    uint64_t addr = req->getAddress();
    switch (req->getcmd()) {
        case Request::Command::Read:
            for (uint64_t offset=0; offset<req->getSize(); offset++) {
                //if (dataArray.find(addr+offset) == dataArray.end()) {
                //    dataArray[addr+offset] = 0;
                //}
                req->getData().at(offset) = dataArray[addr+offset];
            }
            request_queue.push(req);
            break;
        case Request::Command::Write:
            for (uint64_t offset=0; offset<req->getSize(); offset++)
                dataArray[addr+offset] = req->getData().at(offset);
            request_queue.push(req);
            break;
        case Request::Command::ReadResp:
        case Request::Command::WriteResp:
        default:
            std::cerr << "[Error] Unknown command type!" << std::endl;
            break;
    }
}

SimpleMem::Request* SimpleMem::popRequest() {
    if (request_queue.size()==0) {
        return NULL;
    }
    auto req = request_queue.front();
    request_queue.pop();
    return req;
}
