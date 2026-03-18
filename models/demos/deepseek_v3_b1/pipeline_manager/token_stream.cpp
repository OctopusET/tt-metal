#include "models/demos/deepseek_v3_b1/pipeline_manager/token_stream.hpp"

#include <algorithm>

namespace models::demos::deepseek_v3_b1::pipeline_manager {

TokenStream::TokenStream(std::ostream& output_stream) : output_stream_(output_stream) {}

void TokenStream::emit_ready(const std::string& component_name) {
    output_stream_ << "READY\t" << sanitize_field(component_name) << std::endl;
}

void TokenStream::emit_token(const std::string& request_id, size_t token_index, uint32_t token_id) {
    output_stream_ << "TOKEN\t" << sanitize_field(request_id) << "\t" << token_index << "\t" << token_id << std::endl;
}

void TokenStream::emit_complete(const RequestState& request) {
    output_stream_ << "COMPLETE\t" << sanitize_field(request.request_id) << "\t" << request.generated_token_ids.size()
                   << "\t" << join_generated_tokens(request) << std::endl;
}

void TokenStream::emit_error(const std::string& request_id, const std::string& message) {
    output_stream_ << "ERROR\t" << sanitize_field(request_id) << "\t" << sanitize_field(message) << std::endl;
}

std::string TokenStream::sanitize_field(const std::string& value) {
    std::string sanitized = value;
    std::replace(sanitized.begin(), sanitized.end(), '\t', ' ');
    std::replace(sanitized.begin(), sanitized.end(), '\n', ' ');
    std::replace(sanitized.begin(), sanitized.end(), '\r', ' ');
    return sanitized;
}

}  // namespace models::demos::deepseek_v3_b1::pipeline_manager
