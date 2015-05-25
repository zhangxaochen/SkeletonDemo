set (_projname "orbbec_skeleton")

find_package(OpenCV REQUIRED)
add_definitions(-DLINCCCC_DEBUG)

set(${_projname}_SOURCES
  SkeletonPlugin.h
  SkeletonPlugin.cpp
  SkeletonStream.h
  SkeletonTracker.h
  SkeletonTracker.cpp
  
  bodyPartRecognizer.h
  CapgSkeleton.h
  denseDefine.h
  depthFeature.h
  depthSample.h
  logCat.h
  Macros.h
  Vector3.h
  bodyPartRecognizer.cpp
  bprTree.cpp
  depthFeature.cpp
  depthSample.cpp
  logCat.cpp
  )

add_library(${_projname} SHARED ${${_projname}_SOURCES})

target_link_libraries(${_projname} ${SENSEKIT_LIBRARIES} ${OpenCV_LIBS})

include_directories(${SENSEKIT_INCLUDE_DIR})
link_directories(${OpenCV_LIB_DIR})

set_target_properties(${_projname} PROPERTIES FOLDER "${SAMPLE_DIR_FOLDER}plugins")

add_sensekit_sdk_dependency(${_projname})