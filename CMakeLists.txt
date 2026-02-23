cmake_minimum_required(VERSION 3.10)
project(ObjectRecognition)
set(CMAKE_CXX_STANDARD 17)
set(OpenCV_DIR "C:/msys64/ucrt64/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)
include_directories(include ${OpenCV_INCLUDE_DIRS})
add_executable(objectrec
    src/main.cpp
    src/threshold.cpp
    src/morphology.cpp
    src/segmentation.cpp
    src/features.cpp
    src/classifier.cpp
)
target_link_libraries(objectrec ${OpenCV_LIBS})
