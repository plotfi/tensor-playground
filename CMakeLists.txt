# tensara-kernels build.
#
# Replaces the hand-written Makefiles. The key fix over the old pattern rule:
# each kernel/test executable lists BOTH the harness and its solutions-cuda/*.cu
# as sources, so editing a solution (e.g. huber-loss.cu) actually triggers a
# rebuild — and nvcc's generated header deps rebuild on tensor-lib / kernel-
# implementation changes too.
#
# Usage:
#   cmake -S . -B build                 # configure (add -G Ninja if you have it)
#   cmake --build build -j              # build everything (harnesses + tests)
#   cmake --build build -j -t harnesses # just the benchmark harnesses
#   ctest --test-dir build -j          # run the correctness suite
#   ctest --test-dir build -R huber    # run one test (regex on kernel name)
#
# Binaries land in build/bin/<name>.exe (same layout the old scripts expect, so
# run_all.sh still works).

cmake_minimum_required(VERSION 3.24)
project(kernel_benchmarks LANGUAGES CXX)

# --- standards / flags (match the old NVCCFLAGS: -O2 -std=c++17) -------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# --- optional CUDA support ---------------------------------------------------
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    set(HAVE_CUDA TRUE)
else()
    message(STATUS "CUDA not found — CUDA harnesses and tests will be skipped")
    set(HAVE_CUDA FALSE)
endif()

# Binaries -> build/bin, with a .exe suffix so run_all.sh's glob keeps working.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(REPO ${CMAKE_CURRENT_SOURCE_DIR})
# Harnesses/solutions use relative includes, but adding these makes the shared
# headers resolvable regardless of the including file's location.
include_directories(${REPO}/tensor-lib ${REPO}/kernel-implementation)

# A solution file that is still the fill-in stub (its test can build but will
# fail at runtime, so we mark such tests DISABLED — mirroring run_tests.sh).
function(_is_stub path out)
    set(${out} FALSE PARENT_SCOPE)
    if(EXISTS ${path})
        file(READ ${path} _c)
        string(FIND "${_c}" "// TODO: implement" _p)
        if(NOT _p EQUAL -1)
            set(${out} TRUE PARENT_SCOPE)
        endif()
    endif()
endfunction()

# ============================================================================
# CUDA benchmark harnesses:  kernel-harnesses/<k>.cu + solutions-cuda/<k>.cu
# ============================================================================
if(HAVE_CUDA)
add_custom_target(harnesses COMMENT "All CUDA benchmark harnesses")

file(GLOB HARNESS_SRCS CONFIGURE_DEPENDS ${REPO}/kernel-harnesses/*.cu)
foreach(h ${HARNESS_SRCS})
    get_filename_component(k ${h} NAME_WE)
    set(sol ${REPO}/solutions-cuda/${k}.cu)
    set(srcs ${h})
    if(EXISTS ${sol})
        list(APPEND srcs ${sol})     # <-- solution is a real source => rebuilds on edit
    endif()
    add_executable(${k} ${srcs})
    set_target_properties(${k} PROPERTIES SUFFIX ".exe")
    add_dependencies(harnesses ${k})
endforeach()

enable_testing()

# ============================================================================
# Correctness tests (CTest). Each executable is a test source + the kernel's
# solutions-cuda/<k>.cu, some compiled with a -D flag. Test name = kernel name.
# ============================================================================
add_custom_target(tests COMMENT "All CUDA correctness tests")
set(TDIR ${REPO}/tests)

# add_kernel_test(<kernel> <test-source> [DEFS...])
function(add_kernel_test kernel testsrc)
    set(tgt test-${kernel})
    if(TARGET ${tgt})
        return()   # already added (e.g. glob + explicit); explicit wins first
    endif()
    set(sol ${REPO}/solutions-cuda/${kernel}.cu)
    set(srcs ${TDIR}/${testsrc})
    if(EXISTS ${sol})
        list(APPEND srcs ${sol})
    endif()
    add_executable(${tgt} ${srcs})
    set_target_properties(${tgt} PROPERTIES SUFFIX ".exe")
    if(ARGN)
        target_compile_definitions(${tgt} PRIVATE ${ARGN})
    endif()
    add_dependencies(tests ${tgt})
    add_test(NAME ${kernel} COMMAND ${tgt})
    _is_stub(${sol} stub)
    if(stub)
        set_tests_properties(${kernel} PROPERTIES DISABLED TRUE)
    endif()
endfunction()

# --- grouped / flagged tests (must be registered before the glob) -----------
# Activations: one shared source, a -DACT_* per kernel.
set(_acts relu:ACT_RELU elu:ACT_ELU leaky-relu:ACT_LEAKY_RELU swish:ACT_SWISH
          gelu:ACT_GELU selu:ACT_SELU sigmoid:ACT_SIGMOID soft-plus:ACT_SOFTPLUS
          tanh:ACT_TANH hard-sigmoid:ACT_HARD_SIGMOID)
foreach(p ${_acts})
    string(REPLACE ":" ";" kv ${p})
    list(GET kv 0 k)
    list(GET kv 1 f)
    add_kernel_test(${k} test-activation.cu ${f})
endforeach()

# Dimension reductions.
add_kernel_test(sum-dim     dim-reduce.cu OP_SUM)
add_kernel_test(mean-dim    dim-reduce.cu OP_MEAN)
add_kernel_test(max-dim     dim-reduce.cu OP_MAX)
add_kernel_test(min-dim     dim-reduce.cu OP_MIN)
add_kernel_test(product-dim dim-reduce.cu OP_PROD)
# Arg reductions.
add_kernel_test(argmax argreduce.cu)
add_kernel_test(argmin argreduce.cu ARG_MIN)
# Scalar 1D losses. (hinge-loss has its own standalone test-hinge-loss.cu, which
# the test-*.cu glob below picks up automatically.)
add_kernel_test(huber-loss loss-reduce.cu L_HUBER)
add_kernel_test(kl-loss    loss-reduce.cu L_KL)
# Triangular matmul.
add_kernel_test(upper-trig-matmul trig-matmul.cu TRIG_UPPER)
add_kernel_test(lower-trig-matmul trig-matmul.cu TRIG_LOWER)
# Pooling (avg + max, 1/2/3D).
foreach(d 1 2 3)
    add_kernel_test(avg-pool-${d}d avg-pool.cu POOL_DIM=${d})
    add_kernel_test(max-pool-${d}d max-pool.cu POOL_DIM=${d})
endforeach()

# --- simple 1-source tests: tests/test-<k>.cu + solutions-cuda/<k>.cu -------
# Everything not handled above. test-activation.cu is shared, so skip it.
file(GLOB TEST_SRCS CONFIGURE_DEPENDS ${TDIR}/test-*.cu)
foreach(ts ${TEST_SRCS})
    get_filename_component(base ${ts} NAME_WE)   # test-<k>
    if(base STREQUAL "test-activation")
        continue()
    endif()
    string(REGEX REPLACE "^test-" "" kernel ${base})
    add_kernel_test(${kernel} ${base}.cu)
endforeach()

endif() # HAVE_CUDA

# ============================================================================
# Metal backend (macOS only). Each solutions-metal/<k>.cpp embeds its shader
# source via the METAL_KERNEL macro (raw string literal compiled at runtime).
# Shared .metal.h headers are staged next to the binaries so the runtime
# include inliner can resolve them.
# ============================================================================
if(APPLE)
    set(METAL_CPP ${REPO}/metal-cpp CACHE PATH "Path to metal-cpp headers")
    set(METAL_OUT ${CMAKE_BINARY_DIR}/metal)
    file(MAKE_DIRECTORY ${METAL_OUT})

    # Stage shared shader headers once.
    file(GLOB METAL_HDRS ${REPO}/kernel-implementation/*.metal.h)
    foreach(hdr ${METAL_HDRS})
        get_filename_component(hn ${hdr} NAME)
        add_custom_command(OUTPUT ${METAL_OUT}/${hn}
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${hdr} ${METAL_OUT}/${hn}
            DEPENDS ${hdr})
        list(APPEND METAL_HDR_OUT ${METAL_OUT}/${hn})
    endforeach()
    add_custom_target(metal-headers-stage DEPENDS ${METAL_HDR_OUT})

    add_custom_target(metal COMMENT "All Metal harnesses")
    file(GLOB _mh CONFIGURE_DEPENDS ${REPO}/kernel-harnesses/*.cu)
    foreach(h ${_mh})
        get_filename_component(k ${h} NAME_WE)
        if(NOT EXISTS ${REPO}/solutions-metal/${k}.cpp)
            continue()
        endif()
        add_executable(metal-${k}
            ${REPO}/kernel-harnesses/${k}.cu
            ${REPO}/solutions-metal/${k}.cpp)
        set_source_files_properties(
            ${REPO}/kernel-harnesses/${k}.cu ${REPO}/solutions-metal/${k}.cpp
            PROPERTIES LANGUAGE CXX)
        target_compile_options(metal-${k} PRIVATE -x c++)
        target_compile_definitions(metal-${k} PRIVATE TENSOR_METAL)
        set_source_files_properties(${REPO}/solutions-metal/${k}.cpp
            PROPERTIES COMPILE_DEFINITIONS TENSOR_METAL_IMPL)
        target_include_directories(metal-${k} PRIVATE ${METAL_CPP})
        target_link_libraries(metal-${k} PRIVATE "-framework Metal" "-framework Foundation")
        set_target_properties(metal-${k} PROPERTIES
            OUTPUT_NAME ${k}
            RUNTIME_OUTPUT_DIRECTORY ${METAL_OUT})
        add_dependencies(metal-${k} metal-headers-stage)
        add_dependencies(metal metal-${k})
    endforeach()
endif()
