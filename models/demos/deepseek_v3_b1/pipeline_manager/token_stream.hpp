#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>

#include "models/demos/deepseek_v3_b1/pipeline_manager/request_state.hpp"

namespace models::demos::deepseek_v3_b1::pipeline_manager {

class TokenStream {
public:
    explicit TokenStream(std::ostream& output_stream);

    void emit_ready(const std::string& component_name);
    void emit_token(const std::string& request_id, size_t token_index, uint32_t token_id);
    void emit_complete(const RequestState& request);
    void emit_error(const std::string& request_id, const std::string& message);

private:
    std::ostream& output_stream_;

    static std::string sanitize_field(const std::string& value);
};

}  // namespace models::demos::deepseek_v3_b1::pipeline_manager
