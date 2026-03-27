cmake_minimum_required(VERSION 3.22.1)
project(flashmoe_android)

add_library(
    flashmoe_android
    SHARED
    infer_android.cpp
    vulkan_backend.cpp
    jni/com_flashmoe_FlashMoeNative.cpp
)

target_include_directories(
    flashmoe_android
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

find_library(log-lib log)

target_link_libraries(
    flashmoe_android
    ${log-lib}
)
