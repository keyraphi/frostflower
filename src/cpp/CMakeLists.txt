# --- backend (CUDA) ---------------------------------------------------------
add_library(frostflower_backend STATIC frostflower_backend.cu)
set_target_properties(
    frostflower_backend
    PROPERTIES POSITION_INDEPENDENT_CODE ON
)

set(FROSTFLOWER_CUDA_FLAGS
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
)
target_compile_options(frostflower_backend PRIVATE ${FROSTFLOWER_CUDA_FLAGS})
target_include_directories(
    frostflower_backend
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(frostflower_backend PUBLIC CUDA::cudart)

# --- Python module ----------------------------------------------------------
nanobind_add_module(frostflower_module
    frostflower.cpp
)

nanobind_add_stub(
    frostflower_module_stub
    MODULE frostflower_module
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/../frostflower/frostflower_module.pyi
    DEPENDS frostflower_module
    PYTHON_PATH $<TARGET_FILE_DIR:frostflower_module> ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(
    frostflower_module
    PRIVATE frostflower_backend CUDA::cudart
)
target_include_directories(
    frostflower_module
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

# --- install ----------------------------------------------------------------
if(UNIX)
    set_target_properties(frostflower_module PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()
install(TARGETS frostflower_module LIBRARY DESTINATION .)
