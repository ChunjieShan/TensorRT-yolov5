cmake_minimum_required(VERSION 3.10)
project(yolov5)

set(CMAKE_CXX_STANDARD 14)

# cuda
find_package(CUDA REQUIRED)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# tensorrt
include_directories(/home/rick/local_install/TensorRT/include)
link_directories(/home/rick/local_install/TensorRT/lib)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(yolov5 main.cpp)

target_link_libraries(yolov5 nvinfer)
target_link_libraries(yolov5 nvonnxparser)
target_link_libraries(yolov5 cudart)
target_link_libraries(yolov5 ${OpenCV_LIBS})