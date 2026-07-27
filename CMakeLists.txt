cmake_minimum_required(VERSION 3.20)

project(edge_dit VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release MinSizeRel RelWithDebInfo)
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_definitions(GGML_MAX_NAME=128)

find_package(Threads REQUIRED)

option(ED_BUILD_EXAMPLES "Build examples" ON)
option(ED_BUILD_SHARED_LIBS "Build edge-dit as a shared library" OFF)
option(ED_ENABLE_CUDNN_SDPA "Enable experimental cuDNN SDPA backend for CUDA flash attention" OFF)
option(ED_ENABLE_CUDA_NORM "Enable edge CUDA norm fast paths for ggml CUDA backend" ON)
option(ED_ENABLE_CUDA_ROPE "Enable edge CUDA RoPE fast paths for ggml CUDA backend" ON)
option(ED_ENABLE_CUDA_MODULATION "Enable edge CUDA fused modulation fast paths for ggml CUDA backend" ON)
option(ED_ENABLE_CUDA_SAGE_ATTN "Enable SageAttention2-style INT8-QK + F16-PV fused attention for ggml CUDA backend" OFF)
option(ED_ENABLE_PARALLEL "Enable edge-dit lightweight parallel runtime" ON)
option(ED_ENABLE_NCCL "Enable NCCL communication backend" OFF)
option(ED_ENABLE_MPI "Enable MPI bootstrap for distributed process groups" OFF)
option(ED_DEBUG_SP_COMM "Enable debug sequence-parallel communication markers" OFF)
option(ED_BENCHMARK_MARKERS "Emit stdout phase markers for the benchmark harness (component-level attribution)" OFF)
option(ED_FETCH_CUDNN_FRONTEND "Download cudnn-frontend during configure when the vendored copy is absent" OFF)
set(NCCL_ROOT "" CACHE PATH "Path to NCCL installation")
set(CUDNN_ROOT "" CACHE PATH "Path to cuDNN installation")
set(CUDNN_RUNTIME_LIBRARY_DIRS "" CACHE STRING "Extra runtime library directories needed by cuDNN, separated by semicolons")

option(ED_GGML_CUDA   "Enable ggml CUDA backend" OFF)
option(ED_GGML_METAL  "Enable ggml Metal backend" OFF)
option(ED_GGML_VULKAN "Enable ggml Vulkan backend" OFF)
option(ED_GGML_OPENCL "Enable ggml OpenCL backend" OFF)
option(ED_GGML_SYCL   "Enable ggml SYCL backend" OFF)
option(ED_GGML_HIP    "Enable ggml HIP backend" OFF)

if(ED_ENABLE_NCCL)
    find_package(CUDAToolkit REQUIRED)

    set(_ED_NCCL_HINTS)
    if(NCCL_ROOT)
        list(APPEND _ED_NCCL_HINTS "${NCCL_ROOT}")
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/nccl")
        list(APPEND _ED_NCCL_HINTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/nccl")
    endif()

    find_path(ED_NCCL_INCLUDE_DIR
        NAMES nccl.h
        HINTS ${_ED_NCCL_HINTS}
        PATH_SUFFIXES include
    )
    find_library(ED_NCCL_LIBRARY
        NAMES nccl
        HINTS ${_ED_NCCL_HINTS}
        PATH_SUFFIXES lib lib64
    )

    if(NOT ED_NCCL_INCLUDE_DIR OR NOT ED_NCCL_LIBRARY)
        message(FATAL_ERROR "ED_ENABLE_NCCL=ON requires NCCL. Set NCCL_ROOT or provide third_party/nccl.")
    endif()

    file(STRINGS "${ED_NCCL_INCLUDE_DIR}/nccl.h" _ED_NCCL_VERSION_LINES
        REGEX "^#define NCCL_(MAJOR|MINOR|PATCH) +[0-9]+")
    foreach(_ED_NCCL_VERSION_LINE IN LISTS _ED_NCCL_VERSION_LINES)
        if(_ED_NCCL_VERSION_LINE MATCHES "^#define NCCL_MAJOR +([0-9]+)")
            set(ED_NCCL_VERSION_MAJOR "${CMAKE_MATCH_1}")
        elseif(_ED_NCCL_VERSION_LINE MATCHES "^#define NCCL_MINOR +([0-9]+)")
            set(ED_NCCL_VERSION_MINOR "${CMAKE_MATCH_1}")
        elseif(_ED_NCCL_VERSION_LINE MATCHES "^#define NCCL_PATCH +([0-9]+)")
            set(ED_NCCL_VERSION_PATCH "${CMAKE_MATCH_1}")
        endif()
    endforeach()
    if(DEFINED ED_NCCL_VERSION_MAJOR)
        set(ED_NCCL_VERSION "${ED_NCCL_VERSION_MAJOR}.${ED_NCCL_VERSION_MINOR}.${ED_NCCL_VERSION_PATCH}")
    else()
        set(ED_NCCL_VERSION "unknown")
    endif()
endif()

if(ED_ENABLE_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
endif()

if(ED_ENABLE_CUDNN_SDPA OR ((ED_ENABLE_CUDA_NORM OR ED_ENABLE_CUDA_ROPE OR ED_ENABLE_CUDA_MODULATION OR ED_ENABLE_CUDA_SAGE_ATTN) AND ED_GGML_CUDA))
    enable_language(CUDA)
endif()

if(ED_ENABLE_CUDNN_SDPA)
    find_package(CUDAToolkit REQUIRED)
    include(FetchContent)

    if(CUDNN_ROOT)
        set(CUDNN_PATH "${CUDNN_ROOT}" CACHE PATH "Path to cuDNN installation" FORCE)
    elseif(DEFINED ENV{CONDA_PREFIX})
        file(GLOB _ED_CONDA_CUDNN_DIRS "$ENV{CONDA_PREFIX}/lib/python*/site-packages/nvidia/cudnn")
        if(_ED_CONDA_CUDNN_DIRS)
            list(GET _ED_CONDA_CUDNN_DIRS 0 _ED_CONDA_CUDNN_DIR)
            set(CUDNN_PATH "${_ED_CONDA_CUDNN_DIR}" CACHE PATH "Path to cuDNN installation" FORCE)
        endif()
    endif()

    set(ED_CUDNN_RUNTIME_LIBRARY_DIRS ${CUDNN_RUNTIME_LIBRARY_DIRS})
    if(CUDNN_PATH)
        list(APPEND ED_CUDNN_RUNTIME_LIBRARY_DIRS "${CUDNN_PATH}/lib")
        get_filename_component(_ED_CUDNN_PARENT_DIR "${CUDNN_PATH}" DIRECTORY)
        foreach(_ED_CUDNN_PY_LIB_DIR IN ITEMS cuda_nvrtc/lib cuda_runtime/lib cudnn/lib)
            if(EXISTS "${_ED_CUDNN_PARENT_DIR}/${_ED_CUDNN_PY_LIB_DIR}")
                list(APPEND ED_CUDNN_RUNTIME_LIBRARY_DIRS "${_ED_CUDNN_PARENT_DIR}/${_ED_CUDNN_PY_LIB_DIR}")
            endif()
        endforeach()
    endif()
    if(ED_CUDNN_RUNTIME_LIBRARY_DIRS)
        list(REMOVE_DUPLICATES ED_CUDNN_RUNTIME_LIBRARY_DIRS)
    endif()

    set(CUDNN_FRONTEND_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
    set(CUDNN_FRONTEND_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(CUDNN_FRONTEND_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
    if(NOT TARGET cudnn_frontend)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cudnn-frontend/CMakeLists.txt")
            set(ED_CUDNN_FRONTEND_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cudnn-frontend")
            add_subdirectory("${ED_CUDNN_FRONTEND_SOURCE_DIR}")
        elseif(ED_FETCH_CUDNN_FRONTEND)
            FetchContent_Declare(
                cudnn_frontend
                GIT_REPOSITORY https://github.com/NVIDIA/cudnn-frontend.git
                GIT_TAG 07b368a
            )
            FetchContent_MakeAvailable(cudnn_frontend)
            set(ED_CUDNN_FRONTEND_SOURCE_DIR "${cudnn_frontend_SOURCE_DIR}")
        else()
            message(FATAL_ERROR
                "ED_ENABLE_CUDNN_SDPA=ON requires third_party/cudnn-frontend or ED_FETCH_CUDNN_FRONTEND=ON. "
                "Run git submodule update --init --recursive if the dependency is vendored in this checkout.")
        endif()
    endif()
    include("${ED_CUDNN_FRONTEND_SOURCE_DIR}/cmake/cuDNN.cmake")
endif()

if(MSVC)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:C>:/MP>
        $<$<COMPILE_LANGUAGE:CXX>:/MP>
    )
endif()

# Keep ggml clean: do not build ggml's own tests/examples.
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

if(ED_GGML_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
endif()

if(ED_GGML_METAL)
    set(GGML_METAL ON CACHE BOOL "" FORCE)
endif()

if(ED_GGML_VULKAN)
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
endif()

if(ED_GGML_OPENCL)
    set(GGML_OPENCL ON CACHE BOOL "" FORCE)
endif()

if(ED_GGML_SYCL)
    set(GGML_SYCL ON CACHE BOOL "" FORCE)
endif()

if(ED_GGML_HIP)
    set(GGML_HIP ON CACHE BOOL "" FORCE)
endif()

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/CMakeLists.txt")
    message(FATAL_ERROR
        "Missing ggml submodule at third_party/ggml. "
        "Run: git submodule update --init --recursive. "
        "GitHub auto-generated Source ZIP archives usually do not include submodule contents; "
        "prefer git clone --recursive or a complete project release source package.")
endif()

add_subdirectory(third_party/ggml)

if(ED_GGML_CUDA)
    target_include_directories(ggml-cuda
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    )
endif()

if(ED_ENABLE_CUDA_NORM AND ED_GGML_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_sources(ggml-cuda
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src/core/backend/ggml/cuda/ed_cuda_norm.cu
    )
    target_compile_definitions(ggml-cuda PRIVATE ED_ENABLE_CUDA_NORM)
    target_include_directories(ggml-cuda
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src/core/backend/ggml/cuda
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/src/ggml-cuda
    )
endif()

if(ED_ENABLE_CUDA_ROPE AND ED_GGML_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_sources(ggml-cuda
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src/core/backend/ggml/cuda/ed_cuda_attention_v_prep.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/src/core/backend/ggml/cuda/ed_cuda_rope.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/src/core/backend/ggml/cuda/ed_cuda_sp_flux.cu
    )
    target_compile_definitions(ggml-cuda PRIVATE ED_ENABLE_CUDA_ROPE)
    target_include_directories(ggml-cuda
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/src/core
            ${CMAKE_CURRENT_SOURCE_DIR}/src/core/backend/ggml/cuda
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/src/ggml-cuda
    )
endif()

if(ED_ENABLE_CUDA_MODULATION AND ED_GGML_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_sources(ggml-cuda
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src/core/backend/ggml/cuda/ed_cuda_modulation.cu
    )
    target_compile_definitions(ggml-cuda PRIVATE ED_ENABLE_CUDA_MODULATION)
    target_include_directories(ggml-cuda
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/src/core
            ${CMAKE_CURRENT_SOURCE_DIR}/src/core/backend/ggml/cuda
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/src/ggml-cuda
    )
endif()

if(ED_ENABLE_CUDA_SAGE_ATTN AND ED_GGML_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_sources(ggml-cuda
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src/core/backend/ggml/cuda/ed_cuda_sage_attn.cu
    )
    target_compile_definitions(ggml-cuda PRIVATE ED_ENABLE_CUDA_SAGE_ATTN)
    target_include_directories(ggml-cuda
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/src/core
            ${CMAKE_CURRENT_SOURCE_DIR}/src/core/backend/ggml/cuda
            ${CMAKE_CURRENT_SOURCE_DIR}/src/core/backend/ggml/cuda/sage
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/src/ggml-cuda
    )
endif()

if(ED_ENABLE_CUDNN_SDPA)
    target_sources(ggml-cuda
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src/core/backend/cuDNN/ed_cudnn_conv2d.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/src/core/backend/cuDNN/ed_cudnn_conv3d.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/src/core/backend/cuDNN/ed_cudnn_sdpa.cu
    )
    target_compile_definitions(ggml-cuda PRIVATE ED_ENABLE_CUDNN_SDPA ED_ENABLE_CUDNN_CONV2D ED_ENABLE_CUDNN_CONV3D)
    target_include_directories(ggml-cuda
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src/core/backend/cuDNN
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/src/ggml-cuda
            ${ED_CUDNN_FRONTEND_SOURCE_DIR}/include
    )
    target_link_libraries(ggml-cuda
        PRIVATE
            cudnn_frontend
            CUDNN::cudnn_all
            CUDA::nvrtc
            ${CMAKE_DL_LIBS}
    )
    if(ED_CUDNN_RUNTIME_LIBRARY_DIRS)
        string(REPLACE ";" ":" _ED_CUDNN_RUNTIME_LIBRARY_PATHS "${ED_CUDNN_RUNTIME_LIBRARY_DIRS}")
        target_compile_definitions(ggml-cuda PRIVATE ED_CUDNN_RUNTIME_LIBRARY_DIRS="${_ED_CUDNN_RUNTIME_LIBRARY_PATHS}")
        set_property(TARGET ggml-cuda APPEND PROPERTY BUILD_RPATH ${ED_CUDNN_RUNTIME_LIBRARY_DIRS})
    endif()
endif()

file(GLOB_RECURSE ED_CORE_SOURCES CONFIGURE_DEPENDS
    "src/core/*.cpp"
    "src/core/*.h"
    "src/core/*.hpp"
)

file(GLOB ED_API_SOURCES CONFIGURE_DEPENDS
    "src/*.cpp"
    "src/*.h"
    "src/*.hpp"
)

file(GLOB ED_UTIL_SOURCES CONFIGURE_DEPENDS
    "src/utils/*.cpp"
    "src/utils/*.h"
    "src/utils/*.hpp"
)

file(GLOB_RECURSE ED_DIT_MODEL_SOURCES CONFIGURE_DEPENDS
    "src/dit_models/*.cpp"
    "src/dit_models/*.h"
    "src/dit_models/*.hpp"
)
list(FILTER ED_DIT_MODEL_SOURCES EXCLUDE REGEX ".*/super_resolution/.*")

set(ED_SOURCES
    ${ED_API_SOURCES}
    ${ED_CORE_SOURCES}
    ${ED_UTIL_SOURCES}
    ${ED_DIT_MODEL_SOURCES}
    "src/utils/model_io/binary_io.h"
    "src/utils/model_io/convert.cpp"
    "src/utils/model_io/convert.h"
    "src/utils/model_io/gguf_io.cpp"
    "src/utils/model_io/gguf_io.h"
    "src/utils/model_io/gguf_reader_ext.h"
    "src/utils/model_io/safetensors_io.cpp"
    "src/utils/model_io/safetensors_io.h"
    "src/utils/model_io/tensor_storage.h"
)

if(NOT ED_ENABLE_NCCL)
    list(FILTER ED_SOURCES EXCLUDE REGEX ".*/src/core/parallel/backends/nccl/.*")
endif()

if(ED_BUILD_SHARED_LIBS)
    add_library(edgedit SHARED ${ED_SOURCES})
    target_compile_definitions(edgedit PRIVATE ED_BUILD_DLL)
    target_compile_definitions(edgedit PUBLIC ED_BUILD_SHARED_LIB)
else()
    add_library(edgedit STATIC ${ED_SOURCES})
endif()

target_include_directories(edgedit
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/src
)

target_link_libraries(edgedit
    PUBLIC
        ggml
)

target_compile_definitions(edgedit
    PRIVATE
        ED_VERSION_STRING="${PROJECT_VERSION}"
        ED_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
        ED_VERSION_MINOR=${PROJECT_VERSION_MINOR}
        ED_VERSION_PATCH=${PROJECT_VERSION_PATCH}
)

if(ED_ENABLE_CUDNN_SDPA)
    target_compile_definitions(edgedit PRIVATE ED_ENABLE_CUDNN_SDPA ED_ENABLE_CUDNN_CONV2D ED_ENABLE_CUDNN_CONV3D)
endif()

if(ED_ENABLE_CUDA_MODULATION AND ED_GGML_CUDA)
    target_compile_definitions(edgedit PRIVATE ED_ENABLE_CUDA_MODULATION)
endif()

if(ED_ENABLE_PARALLEL)
    target_compile_definitions(edgedit PUBLIC ED_ENABLE_PARALLEL)
endif()

if(ED_ENABLE_NCCL)
    target_compile_definitions(edgedit PUBLIC ED_ENABLE_NCCL)
    target_include_directories(edgedit PRIVATE ${ED_NCCL_INCLUDE_DIR})
    target_link_libraries(edgedit PRIVATE CUDA::cudart ${ED_NCCL_LIBRARY})
endif()

if(ED_ENABLE_MPI)
    target_compile_definitions(edgedit PUBLIC ED_ENABLE_MPI)
    target_link_libraries(edgedit PUBLIC MPI::MPI_CXX)
endif()

if(ED_DEBUG_SP_COMM)
    target_compile_definitions(edgedit PRIVATE ED_DEBUG_SP_COMM=1)
endif()
if(ED_BENCHMARK_MARKERS)
    target_compile_definitions(edgedit PRIVATE ED_BENCHMARK_MARKERS=1)
endif()

target_compile_features(edgedit PUBLIC cxx_std_17)

if(ED_BUILD_EXAMPLES)
    add_executable(ed-cli examples/cli/main.cpp)
    target_link_libraries(ed-cli PRIVATE edgedit)
    target_include_directories(ed-cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
    target_compile_features(ed-cli PRIVATE cxx_std_17)
    if(ED_CUDNN_RUNTIME_LIBRARY_DIRS)
        set_property(TARGET ed-cli APPEND PROPERTY BUILD_RPATH ${ED_CUDNN_RUNTIME_LIBRARY_DIRS})
    endif()

    add_executable(ed-sample examples/cli/sample_main.cpp)
    target_link_libraries(ed-sample PRIVATE edgedit)
    target_include_directories(ed-sample PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
    target_compile_features(ed-sample PRIVATE cxx_std_17)
    if(ED_CUDNN_RUNTIME_LIBRARY_DIRS)
        set_property(TARGET ed-sample APPEND PROPERTY BUILD_RPATH ${ED_CUDNN_RUNTIME_LIBRARY_DIRS})
    endif()

    add_executable(ed-convert examples/cli/convert_main.cpp)
    target_link_libraries(ed-convert PRIVATE edgedit)
    target_include_directories(ed-convert PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    )
    target_compile_features(ed-convert PRIVATE cxx_std_17)
    if(ED_CUDNN_RUNTIME_LIBRARY_DIRS)
        set_property(TARGET ed-convert APPEND PROPERTY BUILD_RPATH ${ED_CUDNN_RUNTIME_LIBRARY_DIRS})
    endif()

    add_executable(ed-server
        examples/server/main.cpp
        examples/server/routes.cpp
        examples/server/runtime.cpp
    )
    target_link_libraries(ed-server PRIVATE edgedit Threads::Threads)
    target_include_directories(ed-server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
    target_compile_features(ed-server PRIVATE cxx_std_17)
    if(ED_CUDNN_RUNTIME_LIBRARY_DIRS)
        set_property(TARGET ed-server APPEND PROPERTY BUILD_RPATH ${ED_CUDNN_RUNTIME_LIBRARY_DIRS})
    endif()
endif()

execute_process(
    COMMAND git rev-parse HEAD
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE ED_GIT_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(NOT ED_GIT_COMMIT)
    set(ED_GIT_COMMIT "unknown")
endif()

execute_process(
    COMMAND git -C "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml" rev-parse HEAD
    OUTPUT_VARIABLE ED_GGML_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(NOT ED_GGML_COMMIT)
    set(ED_GGML_COMMIT "unknown")
endif()

file(WRITE "${CMAKE_BINARY_DIR}/build-config.txt"
"edge_dit_version=${PROJECT_VERSION}
git_commit=${ED_GIT_COMMIT}
ggml_commit=${ED_GGML_COMMIT}
cmake_version=${CMAKE_VERSION}
cmake_generator=${CMAKE_GENERATOR}
build_type=${CMAKE_BUILD_TYPE}
build_shared_libs=${ED_BUILD_SHARED_LIBS}
c_compiler=${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}
cxx_compiler=${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}
cuda_compiler=${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERSION}
cuda_architectures=${CMAKE_CUDA_ARCHITECTURES}
ed_build_examples=${ED_BUILD_EXAMPLES}
ed_ggml_cuda=${ED_GGML_CUDA}
ed_enable_nccl=${ED_ENABLE_NCCL}
ed_nccl_version=${ED_NCCL_VERSION}
ed_enable_mpi=${ED_ENABLE_MPI}
ed_mpi_cxx_version=${MPI_CXX_VERSION}
ed_enable_cudnn_sdpa=${ED_ENABLE_CUDNN_SDPA}
ed_enable_cuda_norm=${ED_ENABLE_CUDA_NORM}
ed_enable_cuda_rope=${ED_ENABLE_CUDA_ROPE}
ed_enable_cuda_modulation=${ED_ENABLE_CUDA_MODULATION}
ed_enable_parallel=${ED_ENABLE_PARALLEL}
ed_debug_sp_comm=${ED_DEBUG_SP_COMM}
ed_benchmark_markers=${ED_BENCHMARK_MARKERS}
")

set(ED_PUBLIC_HEADERS include/edge-dit.h)
set_target_properties(edgedit PROPERTIES PUBLIC_HEADER "${ED_PUBLIC_HEADERS}")

install(TARGETS edgedit
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    PUBLIC_HEADER DESTINATION include
)

if(ED_BUILD_EXAMPLES)
    install(TARGETS ed-cli ed-sample ed-server ed-convert RUNTIME DESTINATION bin)
endif()
