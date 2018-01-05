project(conv)
add_executable(conv conv.cpp)
target_link_libraries(conv Athena)

find_package(OpenCV REQUIRED)
target_link_libraries(conv ${OpenCV_LIBS} )
