file(GLOB PPLNN_SAMPLES_ENGINE_DEMO_SRC *.cc)

add_library(pplnn_samples_engine_demo ${PPLNN_SAMPLES_ENGINE_DEMO_SRC})
target_include_directories(pplnn_samples_engine_demo PRIVATE
    ${flatbuffers_SOURCE_DIR}/include)
target_link_libraries(pplnn_samples_engine_demo PUBLIC pplnn_static)
target_compile_features(pplnn_samples_engine_demo PRIVATE cxx_std_11)
