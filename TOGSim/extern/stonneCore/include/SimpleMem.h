#ifndef SIMPLEMEM_H
#define SIMPLEMEM_H

#include <vector>
#include <cstdint>
#include <queue>
#include <iostream>
#include <map>

// For functional modeling
class SimpleMem {
public:
    class Request {
    public:
        enum class Command { Read, Write, ReadResp, WriteResp };
        static constexpr Command Read = Command::Read;
        static constexpr Command Write = Command::Write;
        static constexpr Command ReadResp = Command::ReadResp;
        static constexpr Command WriteResp = Command::WriteResp;

        Request(Command cmd, uint64_t addr, uint32_t size)
            : id(id_seed++), cmd(cmd), addr(addr), size(size) { data.resize(size); }
        Request(Command cmd, uint64_t addr, uint32_t size, const std::vector<uint8_t>& data)
            : id(id_seed++), cmd(cmd), addr(addr), size(size), data(data) {}

        // Getter functions
        Command getcmd() const { return cmd; }
        void setReply();
        uint64_t getAddress() const { return addr; }
        uint64_t getEndAddress() const { return addr + size; }
        size_t getSize() const { return size; }
        std::vector<uint8_t>& getData() { return data; }
        void setPayload(const std::vector<uint8_t>& newPayload) { data = newPayload; }
        void sendInitData(Request* initRequest );
        void getEndAddress(Request* request);

        uint32_t id;
        static uint32_t id_seed;
        Command cmd;
        uint64_t addr;
        size_t size;
        std::vector<uint8_t> data;
        uint64_t request_time = 0;
        uint32_t stonneId = 0;
    };

    SimpleMem();
    ~SimpleMem() {}
    void init(int phase);
    void sendInitData(Request* req);
    void sendRequest(Request* req);
    Request* popRequest();
protected:
    std::map<uint64_t, uint8_t> dataArray;
    std::queue<Request*> request_queue;
};

#endif // SIMPLEMEM_H
