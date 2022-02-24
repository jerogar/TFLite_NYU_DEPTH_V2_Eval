cmake_minimum_required(VERSION 3.16)
project(DepthEst_TfLite C CXX)

set(CMAKE_CXX_STANDARD 14)
set(TFLITE_BUILD_PATH "/media/Data/lib/tensorflow_src")
set(OPENCV_INSTALL_PATH "/usr/local/")

set(TENSORFLOW_SOURCE_DIR "" CACHE PATH
        "Directory that contains the TensorFlow project" )
if(NOT TENSORFLOW_SOURCE_DIR)
  get_filename_component(TENSORFLOW_SOURCE_DIR
          "${TFLITE_BUILD_PATH}" ABSOLUTE)
endif()

add_subdirectory(
        "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite"
        "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite" EXCLUDE_FROM_ALL)

include_directories("${OPENCV_INSTALL_PATH}/include")
include_directories("${OPENCV_INSTALL_PATH}/include/opencv2")
link_directories("${OPENCV_INSTALL_PATH}/lib")

set(OPENCV_PACKAGE opencv_core opencv_imgproc opencv_imgcodecs opencv_highgui)

file(GLOB_RECURSE SRC_DIR CONFIGURE_DEPENDS src/*.cpp src/*.h)
add_executable(DepthEst_TfLite ${SRC_DIR})
target_link_libraries(DepthEst_TfLite tensorflow-lite ${OPENCV_PACKAGE})

