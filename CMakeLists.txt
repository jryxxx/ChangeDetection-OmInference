cmake_minimum_required(VERSION 3.6)
project(ChangeDetetion)

add_compile_options(-std=c++11)
add_definitions(-DENABLE_DVPP_INTERFACE)

set(DDK_PATH /usr/local/Ascend/ascend-toolkit/latest)
set(NPU_HOST_LIB ${DDK_PATH}/runtime/lib64)
set(THIRDPART /usr/local/Ascend/thirdpart/aarch64)

include_directories(

    # ${PROJECT_SOURCE_DIR}
    # ${PROJECT_SOURCE_DIR}/include

    # ONNXRUNTIME
    ${PROJECT_SOURCE_DIR}/third_party/onnxruntime/include

    # OPENCV
    ${PROJECT_SOURCE_DIR}/third_party/opencv/include
    ${PROJECT_SOURCE_DIR}/third_party/opencv/include/opencv4
    ${PROJECT_SOURCE_DIR}/third_party/opencv/include/opencv4/opencv2

    # EIGEN
    ${PROJECT_SOURCE_DIR}/third_party/eigen/include/eigen3

    # ACL
    ${DDK_PATH}/runtime/include
    ${THIRDPART}/include/acllite
)

# add_library(${PROJECT_NAME}_lib SHARED
# src/ChangeDetection.cpp
# )
link_directories(
    ${NPU_HOST_LIB}
    ${NPU_HOST_LIB}/stub
    ${THIRDPART}/lib

    ${PROJECT_SOURCE_DIR}/third_party/onnxruntime/lib
    ${PROJECT_SOURCE_DIR}/third_party/opencv/lib
)

add_executable(${PROJECT_NAME} test.cpp)

# target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}_lib ascendcl acl_dvpp stdc++ acllite opencv_world onnxruntime dl rt)
target_link_libraries(${PROJECT_NAME} ascendcl acl_dvpp stdc++ acllite opencv_world onnxruntime dl rt)