#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "models/demos/deepseek_v3_b1/pipeline_manager/pipeline_manager.hpp"
#include "models/demos/deepseek_v3_b1/pipeline_manager/request_state.hpp"
#include "models/demos/deepseek_v3_b1/pipeline_manager/socket_connector.hpp"
#include "models/demos/deepseek_v3_b1/pipeline_manager/token_stream.hpp"

namespace models::demos::deepseek_v3_b1::pipeline_manager {
namespace {

std::optional<std::string> get_arg_value(const std::vector<std::string>& arguments, const std::string& key) {
    for (size_t idx = 0; idx + 1 < arguments.size(); ++idx) {
        if (arguments[idx] == key) {
            return arguments[idx + 1];
        }
    }
    return std::nullopt;
}

std::string require_arg_value(const std::vector<std::string>& arguments, const std::string& key) {
    const auto value = get_arg_value(arguments, key);
    if (!value.has_value()) {
        throw std::runtime_error("Missing required argument: " + key);
    }
    return value.value();
}

std::vector<uint32_t> parse_token_list(const std::string& csv_tokens) {
    std::vector<uint32_t> token_ids;
    if (csv_tokens.empty()) {
        return token_ids;
    }

    std::stringstream stream(csv_tokens);
    std::string token;
    while (std::getline(stream, token, ',')) {
        if (!token.empty()) {
            token_ids.push_back(static_cast<uint32_t>(std::stoul(token)));
        }
    }
    return token_ids;
}

std::optional<uint32_t> parse_optional_token_id(const std::string& value) {
    if (value == "-1") {
        return std::nullopt;
    }
    return static_cast<uint32_t>(std::stoul(value));
}

std::vector<std::string> split_tab_separated(const std::string& line) {
    std::vector<std::string> parts;
    std::stringstream stream(line);
    std::string part;
    while (std::getline(stream, part, '\t')) {
        parts.push_back(part);
    }
    return parts;
}

int run_writer_mode(const std::vector<std::string>& arguments) {
    const std::string socket_id = require_arg_value(arguments, "--h2d-socket-id");
    const uint32_t page_size_bytes =
        static_cast<uint32_t>(std::stoul(require_arg_value(arguments, "--page-size-bytes")));
    const uint32_t connect_timeout_ms =
        static_cast<uint32_t>(std::stoul(get_arg_value(arguments, "--connect-timeout-ms").value_or("30000")));

    H2DWriterSocket writer(socket_id, page_size_bytes, connect_timeout_ms);
    TokenStream token_stream(std::cout);
    token_stream.emit_ready("writer");

    std::string line;
    while (std::getline(std::cin, line)) {
        const auto parts = split_tab_separated(line);
        if (parts.empty()) {
            continue;
        }
        if (parts[0] == "SHUTDOWN") {
            writer.barrier();
            return 0;
        }
        if (parts.size() != 3 || parts[0] != "WRITE") {
            throw std::runtime_error("Invalid writer command");
        }
        writer.write_token(static_cast<uint32_t>(std::stoul(parts[2])));
    }

    writer.barrier();
    return 0;
}

int run_reader_mode(const std::vector<std::string>& arguments) {
    const std::string socket_id = require_arg_value(arguments, "--d2h-socket-id");
    const uint32_t page_size_bytes =
        static_cast<uint32_t>(std::stoul(require_arg_value(arguments, "--page-size-bytes")));
    const uint32_t connect_timeout_ms =
        static_cast<uint32_t>(std::stoul(get_arg_value(arguments, "--connect-timeout-ms").value_or("30000")));

    D2HReaderSocket reader(socket_id, page_size_bytes, connect_timeout_ms);
    TokenStream token_stream(std::cout);
    token_stream.emit_ready("reader");

    std::string line;
    while (std::getline(std::cin, line)) {
        const auto parts = split_tab_separated(line);
        if (parts.empty()) {
            continue;
        }
        if (parts[0] == "SHUTDOWN") {
            reader.barrier();
            return 0;
        }
        if (parts.size() != 2 || parts[0] != "READ") {
            throw std::runtime_error("Invalid reader command");
        }
        const uint32_t token_id = reader.read_token();
        std::cout << "TOKEN\t" << parts[1] << "\t" << token_id << std::endl;
    }

    reader.barrier();
    return 0;
}

int run_manager_mode(const std::vector<std::string>& arguments, const std::filesystem::path& executable_path) {
    const std::string h2d_socket_id = require_arg_value(arguments, "--h2d-socket-id");
    const std::string d2h_socket_id = require_arg_value(arguments, "--d2h-socket-id");
    const uint32_t page_size_bytes =
        static_cast<uint32_t>(std::stoul(require_arg_value(arguments, "--page-size-bytes")));
    const uint32_t connect_timeout_ms =
        static_cast<uint32_t>(std::stoul(get_arg_value(arguments, "--connect-timeout-ms").value_or("30000")));

    PipelineManager pipeline_manager(
        executable_path, h2d_socket_id, d2h_socket_id, page_size_bytes, connect_timeout_ms);
    TokenStream token_stream(std::cout);
    token_stream.emit_ready("manager");

    const auto one_shot_request_id = get_arg_value(arguments, "--one-shot-request-id");
    if (one_shot_request_id.has_value()) {
        RequestState request = create_request_state(
            one_shot_request_id.value(),
            parse_token_list(require_arg_value(arguments, "--prompt-token-ids")),
            static_cast<uint32_t>(std::stoul(require_arg_value(arguments, "--max-new-tokens"))),
            parse_optional_token_id(get_arg_value(arguments, "--eos-token-id").value_or("-1")));
        pipeline_manager.run_one_shot(request, token_stream);
        pipeline_manager.shutdown();
        return 0;
    }

    pipeline_manager.run_command_loop(std::cin, token_stream);
    return 0;
}

}  // namespace
}  // namespace models::demos::deepseek_v3_b1::pipeline_manager

int main(int argc, char* argv[]) {
    using namespace models::demos::deepseek_v3_b1::pipeline_manager;

    std::vector<std::string> arguments;
    arguments.reserve(static_cast<size_t>(argc > 1 ? argc - 1 : 0));
    for (int idx = 1; idx < argc; ++idx) {
        arguments.emplace_back(argv[idx]);
    }

    try {
        const std::string mode = get_arg_value(arguments, "--mode").value_or("manager");
        const auto executable_path = std::filesystem::absolute(argv[0]);

        if (mode == "writer") {
            return run_writer_mode(arguments);
        }
        if (mode == "reader") {
            return run_reader_mode(arguments);
        }
        if (mode == "manager") {
            return run_manager_mode(arguments, executable_path);
        }
        throw std::runtime_error("Unsupported mode: " + mode);
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return EXIT_FAILURE;
    }
}
