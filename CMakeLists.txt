cmake_minimum_required(VERSION 3.14)
project(Lab6)

set(CMAKE_CXX_STANDARD 17)

# --- Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(Lab6
        main.cpp
        src/algorithms.cpp
        src/algorithms.h
)

target_link_libraries(Lab6 ${OpenCV_LIBS})
