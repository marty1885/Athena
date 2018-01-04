project(Athena)

option(ATHENA_ENABLE_NNPACK "Enable the NNPACK backend in Athena" ON)

set(ATHENA_SRC)
set(ATHENA_LIB)

list(APPEND ATHENA_SRC Athena.cpp Backend.cpp XtensorBackend.cpp)

if(ATHENA_ENABLE_NNPACK)
	list(APPEND ATHENA_SRC NNPACKBackend.cpp)
	list(APPEND ATHENA_LIB pthread nnpack)
endif()

add_library(Athena SHARED ${ATHENA_SRC})
target_link_libraries(Athena ${ATHENA_LIB})
