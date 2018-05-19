project(Athena)

option(ATHENA_ENABLE_NNPACK "Enable the NNPACK backend in Athena" ON)
option(ATHENA_ENABLE_ARRAYFIRE "Enable the ArrayFire backend in Athena" ON)

set(ATHENA_SRC)
set(ATHENA_LIB)

list(APPEND ATHENA_SRC Athena.cpp Tensor.cpp Backend/Backend.cpp Backend/XtensorBackend.cpp NN/Model.cpp NN/Layers.cpp Utils/Archive.cpp)

if(ATHENA_ENABLE_NNPACK)
	list(APPEND ATHENA_SRC Backend/NNPACKBackend.cpp)
	list(APPEND ATHENA_LIB pthread nnpack)
endif()

if(ATHENA_ENABLE_ARRAYFIRE)
	list(APPEND ATHENA_SRC Backend/ArrayFireBackend.cpp)
	list(APPEND ATHENA_LIB af)
endif()

add_library(Athena SHARED ${ATHENA_SRC})
target_link_libraries(Athena ${ATHENA_LIB})

install(TARGETS Athena
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)
install(DIRECTORY Athena/ DESTINATION include/Athena)
