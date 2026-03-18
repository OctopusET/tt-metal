#include "models/demos/deepseek_v3_b1/pipeline_manager/request_state.hpp"

#include <sstream>

namespace models::demos::deepseek_v3_b1::pipeline_manager {

RequestState create_request_state(
    std::string request_id,
    std::vector<uint32_t> prompt_token_ids,
    uint32_t max_new_tokens,
    std::optional<uint32_t> eos_token_id) {
    RequestState request;
    request.request_id = std::move(request_id);
    request.prompt_token_ids = std::move(prompt_token_ids);
    request.max_new_tokens = max_new_tokens;
    request.eos_token_id = eos_token_id;
    return request;
}

void mark_running(RequestState& request) {
    request.status = RequestStatus::Running;
    request.error_message.clear();
}

void mark_completed(RequestState& request) { request.status = RequestStatus::Completed; }

void mark_failed(RequestState& request, const std::string& error_message) {
    request.status = RequestStatus::Failed;
    request.error_message = error_message;
}

std::string join_generated_tokens(const RequestState& request) {
    std::ostringstream output;
    for (size_t idx = 0; idx < request.generated_token_ids.size(); ++idx) {
        if (idx != 0) {
            output << ",";
        }
        output << request.generated_token_ids[idx];
    }
    return output.str();
}

}  // namespace models::demos::deepseek_v3_b1::pipeline_manager
