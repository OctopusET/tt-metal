#include "models/demos/deepseek_v3_b1/pipeline_manager/pipeline_manager.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace models::demos::deepseek_v3_b1::pipeline_manager {
namespace {

std::vector<std::string> split_tab_separated(const std::string& line) {
    std::vector<std::string> parts;
    std::stringstream stream(line);
    std::string part;
    while (std::getline(stream, part, '\t')) {
        parts.push_back(part);
    }
    return parts;
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

class WorkerProcess {
public:
    WorkerProcess(const std::filesystem::path& executable_path, const std::vector<std::string>& arguments) {
        int stdin_pipe[2] = {-1, -1};
        int stdout_pipe[2] = {-1, -1};
        if (pipe(stdin_pipe) != 0 || pipe(stdout_pipe) != 0) {
            throw std::runtime_error("Failed to create worker pipes");
        }

        pid_ = fork();
        if (pid_ < 0) {
            throw std::runtime_error("Failed to fork worker process");
        }

        if (pid_ == 0) {
            dup2(stdin_pipe[0], STDIN_FILENO);
            dup2(stdout_pipe[1], STDOUT_FILENO);
            close(stdin_pipe[0]);
            close(stdin_pipe[1]);
            close(stdout_pipe[0]);
            close(stdout_pipe[1]);

            std::vector<std::string> argv_storage;
            argv_storage.reserve(arguments.size() + 1);
            argv_storage.push_back(executable_path.string());
            for (const auto& argument : arguments) {
                argv_storage.push_back(argument);
            }

            std::vector<char*> argv;
            argv.reserve(argv_storage.size() + 1);
            for (auto& argument : argv_storage) {
                argv.push_back(argument.data());
            }
            argv.push_back(nullptr);
            execv(executable_path.c_str(), argv.data());
            std::perror("execv");
            _exit(127);
        }

        close(stdin_pipe[0]);
        close(stdout_pipe[1]);
        child_stdin_ = fdopen(stdin_pipe[1], "w");
        child_stdout_ = fdopen(stdout_pipe[0], "r");
        if (child_stdin_ == nullptr || child_stdout_ == nullptr) {
            throw std::runtime_error("Failed to open worker stdio handles");
        }
        setvbuf(child_stdin_, nullptr, _IOLBF, 0);
    }

    ~WorkerProcess() {
        if (child_stdin_ != nullptr) {
            fclose(child_stdin_);
        }
        if (child_stdout_ != nullptr) {
            fclose(child_stdout_);
        }
        if (pid_ > 0) {
            int status = 0;
            waitpid(pid_, &status, WNOHANG);
        }
    }

    void write_line(const std::string& line) {
        if (child_stdin_ == nullptr) {
            throw std::runtime_error("Worker stdin is closed");
        }
        if (std::fprintf(child_stdin_, "%s\n", line.c_str()) < 0) {
            throw std::runtime_error("Failed to write command to worker");
        }
        std::fflush(child_stdin_);
    }

    std::string read_line() {
        if (child_stdout_ == nullptr) {
            throw std::runtime_error("Worker stdout is closed");
        }

        char* buffer = nullptr;
        size_t buffer_size = 0;
        const ssize_t line_size = getline(&buffer, &buffer_size, child_stdout_);
        if (line_size < 0) {
            free(buffer);
            throw std::runtime_error("Worker exited before returning a response");
        }

        std::string line(buffer, static_cast<size_t>(line_size));
        free(buffer);
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
            line.pop_back();
        }
        return line;
    }

    void close_stdin() {
        if (child_stdin_ != nullptr) {
            fclose(child_stdin_);
            child_stdin_ = nullptr;
        }
    }

    void wait_for_exit() {
        if (pid_ <= 0) {
            return;
        }

        int status = 0;
        if (waitpid(pid_, &status, 0) < 0) {
            throw std::runtime_error("Failed waiting for worker process");
        }
        pid_ = -1;
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            throw std::runtime_error("Worker process exited with a non-zero status");
        }
    }

private:
    pid_t pid_ = -1;
    FILE* child_stdin_ = nullptr;
    FILE* child_stdout_ = nullptr;
};

class WriterWorkerClient final : public WriterClient {
public:
    WriterWorkerClient(
        const std::filesystem::path& executable_path,
        const std::string& h2d_socket_id,
        uint32_t page_size_bytes,
        uint32_t connect_timeout_ms) :
        process_(
            executable_path,
            {
                "--mode",
                "writer",
                "--h2d-socket-id",
                h2d_socket_id,
                "--page-size-bytes",
                std::to_string(page_size_bytes),
                "--connect-timeout-ms",
                std::to_string(connect_timeout_ms),
            }) {
        const auto ready = process_.read_line();
        const auto parts = split_tab_separated(ready);
        if (parts.size() != 2 || parts[0] != "READY" || parts[1] != "writer") {
            throw std::runtime_error("Writer worker did not report READY");
        }
    }

    void write_token(const std::string& request_id, uint32_t token_id) override {
        process_.write_line("WRITE\t" + request_id + "\t" + std::to_string(token_id));
    }

    void close() override {
        process_.write_line("SHUTDOWN");
        process_.close_stdin();
        process_.wait_for_exit();
    }

private:
    WorkerProcess process_;
};

class ReaderWorkerClient final : public ReaderClient {
public:
    ReaderWorkerClient(
        const std::filesystem::path& executable_path,
        const std::string& d2h_socket_id,
        uint32_t page_size_bytes,
        uint32_t connect_timeout_ms) :
        process_(
            executable_path,
            {
                "--mode",
                "reader",
                "--d2h-socket-id",
                d2h_socket_id,
                "--page-size-bytes",
                std::to_string(page_size_bytes),
                "--connect-timeout-ms",
                std::to_string(connect_timeout_ms),
            }) {
        const auto ready = process_.read_line();
        const auto parts = split_tab_separated(ready);
        if (parts.size() != 2 || parts[0] != "READY" || parts[1] != "reader") {
            throw std::runtime_error("Reader worker did not report READY");
        }
    }

    uint32_t read_token(const std::string& request_id) override {
        process_.write_line("READ\t" + request_id);
        const auto response = process_.read_line();
        const auto parts = split_tab_separated(response);
        if (parts.size() != 3 || parts[0] != "TOKEN" || parts[1] != request_id) {
            throw std::runtime_error("Reader worker returned an invalid response");
        }
        return static_cast<uint32_t>(std::stoul(parts[2]));
    }

    void close() override {
        process_.write_line("SHUTDOWN");
        process_.close_stdin();
        process_.wait_for_exit();
    }

private:
    WorkerProcess process_;
};

RequestState parse_submit_command(const std::vector<std::string>& parts) {
    if (parts.size() != 5) {
        throw std::runtime_error("SUBMIT command must have 5 tab-separated fields");
    }

    return create_request_state(
        parts[1],
        parse_token_list(parts[4]),
        static_cast<uint32_t>(std::stoul(parts[2])),
        parse_optional_token_id(parts[3]));
}

}  // namespace

PipelineManager::PipelineManager(
    std::filesystem::path executable_path,
    std::string h2d_socket_id,
    std::string d2h_socket_id,
    uint32_t page_size_bytes,
    uint32_t connect_timeout_ms) :
    executable_path_(std::move(executable_path)),
    writer_(std::make_unique<WriterWorkerClient>(executable_path_, h2d_socket_id, page_size_bytes, connect_timeout_ms)),
    reader_(
        std::make_unique<ReaderWorkerClient>(executable_path_, d2h_socket_id, page_size_bytes, connect_timeout_ms)) {}

PipelineManager::~PipelineManager() {
    try {
        shutdown();
    } catch (...) {
    }
}

void PipelineManager::run_command_loop(std::istream& input_stream, TokenStream& token_stream) {
    std::string line;
    while (std::getline(input_stream, line)) {
        if (line.empty()) {
            continue;
        }

        const auto parts = split_tab_separated(line);
        if (parts.empty()) {
            continue;
        }

        if (parts[0] == "SHUTDOWN") {
            break;
        }

        if (parts[0] != "SUBMIT") {
            token_stream.emit_error("manager", "Unsupported command: " + parts[0]);
            continue;
        }

        RequestState request = parse_submit_command(parts);
        if (requests_.contains(request.request_id)) {
            token_stream.emit_error(request.request_id, "Duplicate request_id");
            continue;
        }
        if (busy_) {
            token_stream.emit_error(request.request_id, "PipelineManager currently supports one in-flight request");
            continue;
        }

        busy_ = true;
        requests_.emplace(request.request_id, std::move(request));
        auto& stored_request = requests_.at(parts[1]);

        try {
            run_request(stored_request, *writer_, *reader_, token_stream);
        } catch (const std::exception& error) {
            mark_failed(stored_request, error.what());
            token_stream.emit_error(stored_request.request_id, error.what());
        }
        busy_ = false;
    }

    shutdown();
}

void PipelineManager::run_one_shot(RequestState& request, TokenStream& token_stream) {
    if (busy_) {
        throw std::runtime_error("PipelineManager is already busy");
    }

    busy_ = true;
    try {
        run_request(request, *writer_, *reader_, token_stream);
    } catch (...) {
        busy_ = false;
        throw;
    }
    busy_ = false;
}

void PipelineManager::shutdown() {
    if (shutdown_called_) {
        return;
    }
    shutdown_called_ = true;

    if (writer_ != nullptr) {
        writer_->close();
        writer_.reset();
    }
    if (reader_ != nullptr) {
        reader_->close();
        reader_.reset();
    }
}

}  // namespace models::demos::deepseek_v3_b1::pipeline_manager
