#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace models::demos::deepseek_v3_b1::pipeline_manager {

enum class RequestStatus : uint8_t {
    Pending,
    Running,
    Completed,
    Failed,
};

struct RequestState {
    std::string request_id;
    std::vector<uint32_t> prompt_token_ids;
    uint32_t max_new_tokens = 0;
    std::optional<uint32_t> eos_token_id;
    std::vector<uint32_t> generated_token_ids;
    RequestStatus status = RequestStatus::Pending;
    std::string error_message;
};

RequestState create_request_state(
    std::string request_id,
    std::vector<uint32_t> prompt_token_ids,
    uint32_t max_new_tokens,
    std::optional<uint32_t> eos_token_id);

void mark_running(RequestState& request);
void mark_completed(RequestState& request);
void mark_failed(RequestState& request, const std::string& error_message);
std::string join_generated_tokens(const RequestState& request);

}  // namespace models::demos::deepseek_v3_b1::pipeline_manager
