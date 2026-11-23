cmake_minimum_required(VERSION 3.20)
project(cyclescope CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(cyclescope_core STATIC
  cpp/core/job_reader.cpp
  cpp/core/result_writer.cpp
  cpp/core/mechanism_runtime.cpp
  cpp/core/simulator.cpp
  cpp/core/sync_objects.cpp
  cpp/core/async_groups.cpp
  cpp/core/global_sync.cpp
  cpp/core/dispatch.cpp
  cpp/core/instruction_fetch.cpp
  cpp/core/memory_hierarchy.cpp
  cpp/arch/registry.cpp
  cpp/arch/hopper/module.cpp
  cpp/arch/hopper/wgmma.cpp
  cpp/arch/hopper/proxy_fence.cpp
  cpp/arch/blackwell/module.cpp
  cpp/arch/blackwell/umma.cpp
  cpp/arch/blackwell/tmem.cpp
  cpp/arch/blackwell/clc.cpp
  cpp/arch/blackwell/stas.cpp
)
target_include_directories(cyclescope_core PUBLIC cpp)
target_link_libraries(cyclescope_core PUBLIC crypto)   # segment sha256 (loader)

add_library(cyclescope_convert STATIC
  cpp/convert/facts_reader.cpp
  cpp/convert/segment_builder.cpp
  cpp/convert/semantics.cpp
  cpp/convert/axes.cpp
  cpp/convert/traversal.cpp
  cpp/arch/hopper/convert_rules.cpp
  cpp/arch/blackwell/convert_rules.cpp
)
target_include_directories(cyclescope_convert PUBLIC cpp)

add_executable(cyclescope-convert cpp/convert/main.cpp)
target_link_libraries(cyclescope-convert PRIVATE cyclescope_convert)

add_executable(cyclescope-trace-split cpp/convert/trace_split.cpp)

add_library(cyclescope_capture STATIC
  cpp/capture/facts_writer.cpp
  cpp/capture/static_code.cpp
  cpp/capture/trace_reader.cpp
  cpp/capture/poll_folding.cpp
  cpp/capture/control_flow.cpp
  cpp/capture/operand_facts.cpp
  cpp/capture/tensor_map.cpp
  cpp/capture/memory_refs.cpp
  cpp/capture/global_sync.cpp
  cpp/capture/launch_doc.cpp
  cpp/arch/hopper/capture_rules.cpp
  cpp/arch/blackwell/capture_rules.cpp
)
target_include_directories(cyclescope_capture PUBLIC cpp)
target_link_libraries(cyclescope_capture PRIVATE cyclescope_convert)

add_executable(cyclescope-capture-facts cpp/capture/main.cpp)
target_link_libraries(cyclescope-capture-facts PRIVATE cyclescope_capture cyclescope_convert crypto)

add_executable(cyclescope-engine cpp/main.cpp)
target_link_libraries(cyclescope-engine PRIVATE cyclescope_core)

enable_testing()

find_package(Python3 COMPONENTS Interpreter)

add_executable(job_reader_test cpp/tests/job_reader_test.cpp)
target_link_libraries(job_reader_test PRIVATE cyclescope_core)
add_executable(job_validate_test cpp/tests/job_validate_test.cpp)
target_link_libraries(job_validate_test PRIVATE cyclescope_core)
add_executable(mechanism_runtime_test cpp/tests/mechanism_runtime_test.cpp)
target_link_libraries(mechanism_runtime_test PRIVATE cyclescope_core)
add_executable(simulator_test cpp/tests/simulator_test.cpp)
add_executable(memory_hierarchy_test cpp/tests/memory_hierarchy_test.cpp)
target_link_libraries(memory_hierarchy_test PRIVATE cyclescope_core)
target_link_libraries(simulator_test PRIVATE cyclescope_core)

if(Python3_Interpreter_FOUND)
  add_test(NAME make_fixtures
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_SOURCE_DIR}/tests/data/job/make_fixtures.py
      ${CMAKE_BINARY_DIR}/fixtures)
  set_tests_properties(make_fixtures PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/python")
  add_test(NAME make_sync_fixtures
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_SOURCE_DIR}/tests/data/job/make_sync_fixtures.py
      ${CMAKE_BINARY_DIR}/sync-fixtures)
  set_tests_properties(make_sync_fixtures PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/python")
endif()

add_executable(sync_test cpp/tests/sync_test.cpp)
target_link_libraries(sync_test PRIVATE cyclescope_core)
add_test(NAME sync
  COMMAND sync_test ${CMAKE_BINARY_DIR}/sync-fixtures)
set_tests_properties(sync PROPERTIES DEPENDS make_sync_fixtures)

add_test(NAME make_arch_fixtures
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_SOURCE_DIR}/tests/data/job/make_arch_fixtures.py
    ${CMAKE_BINARY_DIR}/arch-fixtures)
set_tests_properties(make_arch_fixtures PROPERTIES
  ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/python")

add_executable(arch_hopper_test cpp/tests/arch_hopper_test.cpp)
target_link_libraries(arch_hopper_test PRIVATE cyclescope_core)
add_test(NAME arch_hopper
  COMMAND arch_hopper_test ${CMAKE_BINARY_DIR}/arch-fixtures)
set_tests_properties(arch_hopper PROPERTIES DEPENDS make_arch_fixtures)

add_test(NAME job_reader
  COMMAND job_reader_test ${CMAKE_BINARY_DIR}/fixtures/job.bin)
set_tests_properties(job_reader PROPERTIES DEPENDS make_fixtures)

add_test(NAME job_validate
  COMMAND job_validate_test ${CMAKE_BINARY_DIR}/fixtures)
set_tests_properties(job_validate PROPERTIES DEPENDS make_fixtures)
add_test(NAME mechanism_runtime
  COMMAND mechanism_runtime_test ${CMAKE_BINARY_DIR}/fixtures/engine.seg)
set_tests_properties(mechanism_runtime PROPERTIES DEPENDS make_fixtures)
add_test(NAME simulator
  COMMAND simulator_test ${CMAKE_BINARY_DIR}/fixtures/engine.seg)
set_tests_properties(simulator PROPERTIES DEPENDS make_fixtures)
add_test(NAME memory_hierarchy
  COMMAND $<TARGET_FILE:memory_hierarchy_test>)
add_test(NAME simulator_unplaced
  COMMAND simulator_test ${CMAKE_BINARY_DIR}/fixtures/engine-unplaced.seg)
set_tests_properties(simulator_unplaced PROPERTIES DEPENDS make_fixtures)

add_executable(instruction_fetch_test cpp/tests/instruction_fetch_test.cpp)
target_link_libraries(instruction_fetch_test PRIVATE cyclescope_core)
add_test(NAME instruction_fetch
  COMMAND $<TARGET_FILE:instruction_fetch_test>)

add_executable(poll_folding_test cpp/tests/poll_folding_test.cpp)
target_link_libraries(poll_folding_test PRIVATE cyclescope_capture cyclescope_convert)
add_test(NAME poll_folding
  COMMAND $<TARGET_FILE:poll_folding_test>)

add_executable(str_util_test cpp/tests/str_util_test.cpp)
target_include_directories(str_util_test PRIVATE cpp)
add_test(NAME str_util
  COMMAND $<TARGET_FILE:str_util_test>)

if(Python3_Interpreter_FOUND)
  add_test(NAME make_resource_fixtures
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_SOURCE_DIR}/tests/data/job/make_resource_fixtures.py
      ${CMAKE_BINARY_DIR}/resource-fixtures)
  set_tests_properties(make_resource_fixtures PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/python")
endif()
add_executable(resource_test cpp/tests/resource_test.cpp)
target_link_libraries(resource_test PRIVATE cyclescope_core)
add_test(NAME resource
  COMMAND resource_test ${CMAKE_BINARY_DIR}/resource-fixtures)
set_tests_properties(resource PROPERTIES DEPENDS make_resource_fixtures)

add_executable(arch_blackwell_test cpp/tests/arch_blackwell_test.cpp)
target_link_libraries(arch_blackwell_test PRIVATE cyclescope_core)
add_test(NAME arch_blackwell
  COMMAND arch_blackwell_test ${CMAKE_BINARY_DIR}/arch-fixtures)
set_tests_properties(arch_blackwell PROPERTIES DEPENDS make_arch_fixtures)

# test binaries keep their asserts in Release builds (after every test
# target exists; a target_compile_options on a not-yet-defined target is
# silently skipped by the TARGET guard)
foreach(t job_reader_test job_validate_test mechanism_runtime_test simulator_test
          memory_hierarchy_test sync_test resource_test instruction_fetch_test
          arch_blackwell_test arch_hopper_test)
  target_compile_options(${t} PRIVATE -UNDEBUG)
endforeach()
