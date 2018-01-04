project(Athena)

add_library(Athena SHARED Athena.cpp Backend.cpp XtensorBackend.cpp)

if(ATHENA_ENABLE_NNPACK)
	target_link_libraries(Athena pthread nnpack)
endif()
