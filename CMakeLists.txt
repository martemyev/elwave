cmake_minimum_required(VERSION 2.8)


project(elwave)


if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build: Debug | Release | PDebug | PRelease" FORCE)
endif()
string(TOUPPER ${CMAKE_BUILD_TYPE} BUILD_TYPE)
message(STATUS "BUILD_TYPE ${BUILD_TYPE}")

add_definitions("-std=c++0x -Wall") # some extra flags
if(BUILD_TYPE STREQUAL "DEBUG" OR BUILD_TYPE STREQUAL "PDEBUG")
  add_definitions("-g")
endif()

#-------------------------------------------------------------------------------
# MFEM
#-------------------------------------------------------------------------------
find_path(MFEM_DIR include/mfem.hpp HINTS ENV MFEM_DIR PATHS $ENV{HOME}/projects/mfem DOC "MFEM directory")
set(MFEM_INCLUDE_DIR ${MFEM_DIR}/include)

unset(MFEM_CONFIG CACHE)
find_path(MFEM_CONFIG config.hpp HINTS ENV MFEM_DIR PATHS ${MFEM_INCLUDE_DIR}/config DOC "MFEM config directory")

list(APPEND MFEM_INCLUDE_DIR ${MFEM_CONFIG})
include_directories(${MFEM_INCLUDE_DIR})

message(STATUS "MFEM_DIR ${MFEM_DIR}")
message(STATUS "MFEM_CONFIG ${MFEM_CONFIG}")
message(STATUS "MFEM_INCLUDE_DIR ${MFEM_INCLUDE_DIR}")

unset(MFEM_LIBRARY CACHE)
find_library(MFEM_LIBRARY mfem HINTS ${MFEM_DIR} PATHS ${MFEM_DIR}/lib PATH_SUFFIXES lib DOC "MFEM library")
if(NOT MFEM_LIBRARY)
  message(FATAL_ERROR "MFEM library was NOT found")
else()
  message(STATUS "MFEM lib ${MFEM_LIBRARY}")
endif()

#-------------------------------------------------------------------------------
# Parallel mode
#-------------------------------------------------------------------------------
if(BUILD_TYPE STREQUAL "PDEBUG" OR BUILD_TYPE STREQUAL "PRELEASE")
  # MPI
  find_package(MPI REQUIRED)
  include_directories(${MPI_CXX_INCLUDE_PATH})
  # Hypre
  find_path(HYPRE_DIR include/HYPRE.h HINTS ENV HYPRE_DIR PATHS "$ENV{HOME}/hypre" DOC "Hypre directory")
  if(EXISTS ${HYPRE_DIR}/include/HYPRE.h)
    set(HYPRE_FOUND YES)
    message(STATUS "Hypre found ${HYPRE_DIR}")
    set(HYPRE_INCLUDES ${HYPRE_DIR})
    find_path(HYPRE_INCLUDE_DIR HYPRE.h HINTS "${HYPRE_DIR}" PATH_SUFFIXES include NO_DEFAULT_PATH)
    list(APPEND HYPRE_INCLUDES ${HYPRE_INCLUDE_DIR})
    include_directories(${HYPRE_INCLUDES})
    find_library(HYPRE_LIBRARY NAMES hypre HYPRE HINTS "${HYPRE_DIR}/lib" DOC "Hypre library")
    message(STATUS "Hypre lib ${HYPRE_LIBRARY}")
    get_filename_component(HYPRE_LIB_DIR ${HYPRE_LIBRARY} DIRECTORY)
    message(STATUS "HYPRE_LIB_DIR ${HYPRE_LIB_DIR}")
    link_directories(${HYPRE_LIB_DIR})
  else()
    set(HYPRE_FOUND NO)
    message(FATAL_ERROR "Hypre is NOT found")
  endif()
  # Metis
  find_path(METIS_DIR include/metis.h HINTS ENV METIS_DIR PATHS "$ENV{HOME}/metis" DOC "Metis directory")
  if(EXISTS ${METIS_DIR}/include/metis.h)
    set(METIS_FOUND YES)
    message(STATUS "Metis found ${METIS_DIR}")
    set(METIS_INCLUDES ${METIS_DIR})
    find_path(METIS_INCLUDE_DIR metis.h HINTS "${METIS_DIR}" PATH_SUFFIXES include NO_DEFAULT_PATH)
    list(APPEND METIS_INCLUDES ${METIS_INCLUDE_DIR})
    include_directories(${METIS_INCLUDES})
    find_library(METIS_LIBRARY NAMES metis METIS HINTS "${METIS_DIR}/lib" DOC "Metis library")
    message(STATUS "Metis lib ${METIS_LIBRARY}")
    string(FIND ${METIS_DIR} "metis-5" IS_METIS_5)
    if(IS_METIS_5)
      set(MFEM_USE_METIS_5 ON)
    else()
      set(MFEM_USE_METIS_5 OFF)
    endif()
    get_filename_component(METIS_LIB_DIR ${METIS_LIBRARY} DIRECTORY)
    message(STATUS "METIS_LIB_DIR ${METIS_LIB_DIR}")
    link_directories(${METIS_LIB_DIR})
  else()
    message(FATAL_ERROR "Metis is NOT found")
    set(METIS_FOUND NO)
  endif()
endif()


find_package(LAPACK REQUIRED)


if(NOT(BUILD_TYPE STREQUAL "RELEASE") AND
   NOT(BUILD_TYPE STREQUAL "PRELEASE"))
  if(NOT(${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC"))
    add_definitions("-fno-inline")
  endif()
endif()


file(GLOB SRC_LIST "${PROJECT_SOURCE_DIR}/src/*.cpp")
file(GLOB HDR_LIST "${PROJECT_SOURCE_DIR}/src/*.hpp")
include_directories("${PROJECT_SOURCE_DIR}/src") # for "config.hpp"


configure_file(
  "${PROJECT_SOURCE_DIR}/config.hpp.in"
  "${PROJECT_SOURCE_DIR}/src/config.hpp")


message("\n--------------- configuration information ----------------")
  message("build type     : " ${BUILD_TYPE})
  message("compiler       : " ${CMAKE_CXX_COMPILER})
  message("hypre dir      : " ${HYPRE_DIR})
  message("hypre lib      : " ${HYPRE_LIBRARIES})
  message("metis dir      : " ${METIS_DIR})
  message("metis lib      : " ${METIS_LIBRARIES})
if(BUILD_TYPE STREQUAL "DEBUG")
  message("compiler flags : " ${CMAKE_CXX_FLAGS_DEBUG})
elseif(BUILD_TYPE STREQUAL "RELWITHDEBINFO")
  message("compiler flags : " ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})
elseif(BUILD_TYPE STREQUAL "RELEASE")
  message("compiler flags : " ${CMAKE_CXX_FLAGS_RELEASE})
endif()


add_executable(${PROJECT_NAME} ${SRC_LIST} ${HDR_LIST})
target_link_libraries(${PROJECT_NAME} ${MFEM_LIBRARY})
target_link_libraries(${PROJECT_NAME} ${LAPACK_LIBRARIES})
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
  target_link_libraries(${PROJECT_NAME} rt)
endif()
if(BUILD_TYPE STREQUAL "PDEBUG" OR BUILD_TYPE STREQUAL "PRELEASE")
  target_link_libraries(${PROJECT_NAME} ${MPI_CXX_LIBRARIES} ${HYPRE_LIBRARY} ${METIS_LIBRARY})
  if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
    target_link_libraries(${PROJECT_NAME} -lmkl -openmp)
  endif()
endif()

