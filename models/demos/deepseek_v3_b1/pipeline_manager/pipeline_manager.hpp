#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>

#include "models/demos/deepseek_v3_b1/pipeline_manager/prefill_decode.hpp"

namespace models::demos::deepseek_v3_b1::pipeline_manager {

class PipelineManager {
public:
    PipelineManager(
        std::filesystem::path executable_path,
        std::string h2d_socket_id,
        std::string d2h_socket_id,
        uint32_t page_size_bytes,
        uint32_t connect_timeout_ms);
    ~PipelineManager();

    void run_command_loop(std::istream& input_stream, TokenStream& token_stream);
    void run_one_shot(RequestState& request, TokenStream& token_stream);
    void shutdown();

private:
    std::filesystem::path executable_path_;
    std::unordered_map<std::string, RequestState> requests_;
    std::unique_ptr<WriterClient> writer_;
    std::unique_ptr<ReaderClient> reader_;
    bool busy_ = false;
    bool shutdown_called_ = false;
};

}  // namespace models::demos::deepseek_v3_b1::pipeline_manager
