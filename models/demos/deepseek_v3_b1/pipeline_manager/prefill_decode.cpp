#include "models/demos/deepseek_v3_b1/pipeline_manager/prefill_decode.hpp"

#include <stdexcept>

namespace models::demos::deepseek_v3_b1::pipeline_manager {

uint32_t run_prefill(RequestState& request, WriterClient& writer, ReaderClient& reader, TokenStream& token_stream) {
    if (request.prompt_token_ids.empty()) {
        throw std::runtime_error("Prompt token list must not be empty");
    }

    uint32_t last_output_token = 0;
    for (size_t idx = 0; idx < request.prompt_token_ids.size(); ++idx) {
        writer.write_token(request.request_id, request.prompt_token_ids[idx]);
        last_output_token = reader.read_token(request.request_id);
    }

    request.generated_token_ids.push_back(last_output_token);
    token_stream.emit_token(request.request_id, 0, last_output_token);
    return last_output_token;
}

void run_decode(
    RequestState& request,
    uint32_t first_generated_token,
    WriterClient& writer,
    ReaderClient& reader,
    TokenStream& token_stream) {
    uint32_t input_token = first_generated_token;
    for (uint32_t step = 0; step + 1 < request.max_new_tokens; ++step) {
        if (request.eos_token_id.has_value() && input_token == request.eos_token_id.value()) {
            break;
        }

        writer.write_token(request.request_id, input_token);
        const uint32_t output_token = reader.read_token(request.request_id);
        request.generated_token_ids.push_back(output_token);
        token_stream.emit_token(request.request_id, request.generated_token_ids.size() - 1, output_token);
        input_token = output_token;
    }
}

void run_request(RequestState& request, WriterClient& writer, ReaderClient& reader, TokenStream& token_stream) {
    if (request.max_new_tokens == 0) {
        throw std::runtime_error("max_new_tokens must be greater than zero");
    }

    mark_running(request);
    const uint32_t first_generated_token = run_prefill(request, writer, reader, token_stream);
    run_decode(request, first_generated_token, writer, reader, token_stream);
    mark_completed(request);
    token_stream.emit_complete(request);
}

}  // namespace models::demos::deepseek_v3_b1::pipeline_manager
