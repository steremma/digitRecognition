cmake_minimum_required (VERSION 2.8.3)
project (digitRecognition)

include_directories (include)
link_directories (include)

set(CMAKE_BUILD_TYPE Release)

file(GLOB SOURCES "src/*.cpp")
add_executable(Main ${SOURCES})

