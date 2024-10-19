cmake_minimum_required(VERSION 3.0)
project(yolov5s_demo)

# using mnn
include_directories(${CMAKE_CURRENT_LIST_DIR}/include)
link_directories(${CMAKE_CURRENT_LIST_DIR}/lib)

# using opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

# yolo_demo
file(GLOB_RECURSE YOLO_SRC ${CMAKE_CURRENT_LIST_DIR}/source/*.cpp)
add_executable(yolo_demo ${YOLO_SRC})
target_link_libraries(yolo_demo ${OpenCV_LIBS} -lMNN)
