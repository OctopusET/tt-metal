set(DEEPSEEK_V3_B1_PIPELINE_MANAGER_SOURCES
    ${PROJECT_SOURCE_DIR}/models/demos/deepseek_v3_b1/pipeline_manager/main.cpp
    ${PROJECT_SOURCE_DIR}/models/demos/deepseek_v3_b1/pipeline_manager/pipeline_manager.cpp
    ${PROJECT_SOURCE_DIR}/models/demos/deepseek_v3_b1/pipeline_manager/prefill_decode.cpp
    ${PROJECT_SOURCE_DIR}/models/demos/deepseek_v3_b1/pipeline_manager/request_state.cpp
    ${PROJECT_SOURCE_DIR}/models/demos/deepseek_v3_b1/pipeline_manager/socket_connector.cpp
    ${PROJECT_SOURCE_DIR}/models/demos/deepseek_v3_b1/pipeline_manager/token_stream.cpp
)

add_executable(deepseek_v3_b1_pipeline_manager ${DEEPSEEK_V3_B1_PIPELINE_MANAGER_SOURCES})
target_link_libraries(
    deepseek_v3_b1_pipeline_manager
    PRIVATE
        Metalium::Metal
        TT::STL
)
target_include_directories(
    deepseek_v3_b1_pipeline_manager
    PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${PROJECT_SOURCE_DIR}/tt_metal/common
        .
)

set_target_properties(
    deepseek_v3_b1_pipeline_manager
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/tools
)

tt_reuse_precompile_headers(deepseek_v3_b1_pipeline_manager TT::CommonPCH)
