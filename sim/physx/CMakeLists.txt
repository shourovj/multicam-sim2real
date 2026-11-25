cmake_minimum_required(VERSION 3.22)
project(p2s CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(DEPS ${CMAKE_CURRENT_SOURCE_DIR}/.deps)
set(PHYSX_ROOT ${DEPS}/physx-src/physx CACHE PATH "PhysX 5.9.1 tree with physx-5.9.1-s2r.patch applied and built")
set(CUDA_RT_DIR ${DEPS}/cuda13/lib/python3.11/site-packages/nvidia/cu13 CACHE PATH "CUDA 13 runtime (lib/libcudart.so.13)")
set(PHYSX_LIB_DIR ${PHYSX_ROOT}/bin/linux.x86_64/release)

add_compile_options(-Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers)

# Core: math, JSON spec, shapes, cameras, X streams. No PhysX, so the checker builds and runs without a GPU.
file(GLOB CORE_SRC CONFIGURE_DEPENDS src/core/*.cpp)
add_library(p2s_core STATIC ${CORE_SRC})
target_include_directories(p2s_core PUBLIC src ${CMAKE_CURRENT_SOURCE_DIR})

# PhysX scene: world, entities, controllers.
file(GLOB PHYSX_SRC CONFIGURE_DEPENDS src/physx/*.cpp src/world/*.cpp src/bodies/*.cpp src/control/*.cpp)
set(VEHICLE_SNIPPETS ${PHYSX_ROOT}/snippets/snippetvehiclecommon)
add_library(p2s_physx STATIC ${PHYSX_SRC}
  ${VEHICLE_SNIPPETS}/base/Base.cpp
  ${VEHICLE_SNIPPETS}/directdrivetrain/DirectDrivetrain.cpp
  ${VEHICLE_SNIPPETS}/physxintegration/PhysXIntegration.cpp)
target_compile_definitions(p2s_physx PUBLIC PX_PHYSX_STATIC_LIB NDEBUG
  P2S_ROBOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/robots")
target_include_directories(p2s_physx PUBLIC ${PHYSX_ROOT}/include ${PHYSX_ROOT}/snippets ${CUDA_RT_DIR}/include)
target_link_directories(p2s_physx PUBLIC ${PHYSX_LIB_DIR})
target_link_libraries(p2s_physx PUBLIC p2s_core
  -Wl,--start-group PhysXVehicle_static_64 PhysXExtensions_static_64 PhysX_static_64 PhysXPvdSDK_static_64
  PhysXCooking_static_64 PhysXCommon_static_64 PhysXFoundation_static_64 -Wl,--end-group pthread dl rt
  # libPhysXGpu_64.so (loaded at runtime) needs the CUDA runtime's symbols already in the process.
  -Wl,--no-as-needed ${CUDA_RT_DIR}/lib/libcudart.so.13 -Wl,--as-needed)

# Checks: reads S and X, writes the report r. No PhysX.
file(GLOB CHECK_SRC CONFIGURE_DEPENDS src/check/*.cpp)
if(CHECK_SRC)
  add_library(p2s_check_lib STATIC ${CHECK_SRC})
  target_link_libraries(p2s_check_lib PUBLIC p2s_core)
endif()

set(CMAKE_BUILD_RPATH ${PHYSX_LIB_DIR} ${CUDA_RT_DIR}/lib)

add_executable(p2s_sim app/sim_main.cpp)
target_link_libraries(p2s_sim PRIVATE p2s_physx)

if(TARGET p2s_check_lib)
  add_executable(p2s_check app/check_main.cpp)
  target_link_libraries(p2s_check PRIVATE p2s_check_lib)
endif()

file(GLOB TEST_SRC CONFIGURE_DEPENDS tests/*.cpp)
if(TEST_SRC)
  add_executable(p2s_tests ${TEST_SRC})
  target_link_libraries(p2s_tests PRIVATE p2s_physx)
  if(TARGET p2s_check_lib)
    target_link_libraries(p2s_tests PRIVATE p2s_check_lib)
  endif()
endif()
