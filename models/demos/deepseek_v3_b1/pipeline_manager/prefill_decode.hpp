#pragma once

#include <cstdint>
#include <string>

#include "models/demos/deepseek_v3_b1/pipeline_manager/request_state.hpp"
#include "models/demos/deepseek_v3_b1/pipeline_manager/token_stream.hpp"

namespace models::demos::deepseek_v3_b1::pipeline_manager {

class WriterClient {
public:
    virtual ~WriterClient() = default;

    virtual void write_token(const std::string& request_id, uint32_t token_id) = 0;
    virtual void close() = 0;
};

class ReaderClient {
public:
    virtual ~ReaderClient() = default;

    virtual uint32_t read_token(const std::string& request_id) = 0;
    virtual void close() = 0;
};

uint32_t run_prefill(RequestState& request, WriterClient& writer, ReaderClient& reader, TokenStream& token_stream);
void run_decode(
    RequestState& request,
    uint32_t first_generated_token,
    WriterClient& writer,
    ReaderClient& reader,
    TokenStream& token_stream);
void run_request(RequestState& request, WriterClient& writer, ReaderClient& reader, TokenStream& token_stream);

}  // namespace models::demos::deepseek_v3_b1::pipeline_manager
