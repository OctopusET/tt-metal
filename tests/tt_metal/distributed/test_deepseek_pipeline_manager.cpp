// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdio>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>

#include "impl/context/metal_context.hpp"
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"

static int g_world_rank = -1;
static int g_world_size = -1;
static int g_test_counter = 0;

namespace tt::tt_metal::distributed {
namespace {

using namespace tt::tt_metal::distributed::multihost;

constexpr uint32_t PAGE_SIZE_BYTES = 64;
constexpr uint32_t FIFO_SIZE_BYTES = 1024;
constexpr uint32_t MAX_NEW_TOKENS = 4;
constexpr uint32_t PROMPT_TOKEN_COUNT = 3;
constexpr uint32_t NUM_TRANSACTIONS = PROMPT_TOKEN_COUNT + (MAX_NEW_TOKENS - 1);
const MeshCoreCoord SOCKET_CORE = {MeshCoordinate(0, 0), CoreCoord(0, 0)};

std::vector<std::string> split_tab_separated(const std::string& line) {
    std::vector<std::string> parts;
    std::stringstream stream(line);
    std::string part;
    while (std::getline(stream, part, '\t')) {
        parts.push_back(part);
    }
    return parts;
}

void run_launcher(const std::shared_ptr<MeshDevice>& mesh_device) {
    const std::string h2d_socket_id = "deepseek_manager_test_h2d_" + std::to_string(g_test_counter);
    const std::string d2h_socket_id = "deepseek_manager_test_d2h_" + std::to_string(g_test_counter);

    auto h2d_socket = H2DSocket(mesh_device, SOCKET_CORE, BufferType::L1, FIFO_SIZE_BYTES, H2DMode::HOST_PUSH);
    h2d_socket.export_descriptor(h2d_socket_id);

    auto d2h_socket = D2HSocket(mesh_device, SOCKET_CORE, FIFO_SIZE_BYTES);
    d2h_socket.export_descriptor(d2h_socket_id);

    auto program = CreateProgram();
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/pcie_socket_loopback.cpp",
        SOCKET_CORE.core_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {
                static_cast<uint32_t>(h2d_socket.get_config_buffer_address()),
                static_cast<uint32_t>(d2h_socket.get_config_buffer_address()),
                PAGE_SIZE_BYTES,
                PAGE_SIZE_BYTES * NUM_TRANSACTIONS,
                1,
                0,
            }});

    auto mesh_workload = MeshWorkload();
    mesh_workload.add_program(MeshCoordinateRange(SOCKET_CORE.device_coord), std::move(program));
    EnqueueMeshWorkload(mesh_device->mesh_command_queue(), mesh_workload, false);
    Finish(mesh_device->mesh_command_queue());
}

void run_manager_connector() {
    const std::string h2d_socket_id = "deepseek_manager_test_h2d_" + std::to_string(g_test_counter);
    const std::string d2h_socket_id = "deepseek_manager_test_d2h_" + std::to_string(g_test_counter);
    const std::string request_id = "request_0";
    const std::string prompt_tokens = "11,22,33";

    std::stringstream command;
    command << PIPELINE_MANAGER_BINARY_PATH << " --mode manager"
            << " --h2d-socket-id " << h2d_socket_id << " --d2h-socket-id " << d2h_socket_id << " --page-size-bytes "
            << PAGE_SIZE_BYTES << " --connect-timeout-ms 30000"
            << " --one-shot-request-id " << request_id << " --prompt-token-ids " << prompt_tokens
            << " --max-new-tokens " << MAX_NEW_TOKENS << " --eos-token-id -1";

    FILE* output_pipe = popen(command.str().c_str(), "r");
    ASSERT_NE(output_pipe, nullptr) << "Failed to launch deepseek_v3_b1_pipeline_manager";

    std::vector<uint32_t> streamed_tokens;
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), output_pipe) != nullptr) {
        std::string line(buffer);
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
            line.pop_back();
        }
        auto parts = split_tab_separated(line);
        ASSERT_FALSE(parts.empty());
        if (parts[0] == "READY") {
            ASSERT_EQ(parts.size(), 2);
            EXPECT_EQ(parts[1], "manager");
            continue;
        }
        if (parts[0] == "TOKEN") {
            ASSERT_EQ(parts.size(), 4);
            EXPECT_EQ(parts[1], request_id);
            streamed_tokens.push_back(static_cast<uint32_t>(std::stoul(parts[3])));
            continue;
        }
        if (parts[0] == "COMPLETE") {
            ASSERT_EQ(parts.size(), 4);
            EXPECT_EQ(parts[1], request_id);
            EXPECT_EQ(static_cast<uint32_t>(std::stoul(parts[2])), MAX_NEW_TOKENS);
            break;
        }
        FAIL() << "Unexpected pipeline manager output line: " << line;
    }

    const int exit_code = pclose(output_pipe);
    ASSERT_EQ(exit_code, 0) << "Pipeline manager exited with a non-zero status";

    const std::vector<uint32_t> expected_tokens = {33, 33, 33, 33};
    EXPECT_EQ(streamed_tokens, expected_tokens);
}

class DeepSeekPipelineManagerFixture : public MeshDeviceFixtureBase {
protected:
    DeepSeekPipelineManagerFixture() : MeshDeviceFixtureBase(Config{.mesh_shape = MeshShape{1, 1}}) {}

    void SetUp() override {
        ASSERT_EQ(g_world_size, 2) << "This test requires exactly 2 MPI ranks";
        rank_ = g_world_rank;
        if (rank_ == 0) {
            MeshDeviceFixtureBase::SetUp();
        }
    }

    void TearDown() override {
        if (rank_ == 0) {
            MeshDeviceFixtureBase::TearDown();
        }
    }

    int rank_ = -1;
};

TEST_F(DeepSeekPipelineManagerFixture, StreamsTokensAcrossExportedSockets) {
    if (rank_ == 0) {
        if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
            GTEST_SKIP() << "Mapping host memory to NOC is not supported on this system";
        }
        run_launcher(mesh_device_);
    } else {
        run_manager_connector();
    }
    ++g_test_counter;
}

}  // namespace
}  // namespace tt::tt_metal::distributed

int main(int argc, char** argv) {
    using namespace tt::tt_metal::distributed::multihost;

    DistributedContext::create(argc, argv);
    const auto& world = DistributedContext::get_current_world();
    g_world_rank = *world->rank();
    g_world_size = *world->size();

    auto local_ctx = world->split(Color(g_world_rank), Key(0));
    DistributedContext::set_current_world(local_ctx);

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
