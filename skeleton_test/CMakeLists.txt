cmake_minimum_required(VERSION 3.2)
set (_projname "skeleton_test")
project(${_projname})

find_package(OpenCV REQUIRED)
add_definitions(-DLINCCCC_DEBUG)

set(${_projname}_SOURCES
	./main.cpp
	
	../SimpleSilhouette.h
	../SimpleSilhouette.cpp

	../sgf_seed/sgf_segment.cpp
	../sgf_seed/Holefilling.h
	../sgf_seed/Node.h
	../sgf_seed/sgf_segment.h
	../sgf_seed/Holefilling.cpp
#	../sgf_seed/main.cpp

	../bodyPartRecognizer.h
	../CapgSkeleton.h
	../denseDefine.h
	../depthFeature.h
	../depthSample.h
	../logCat.h
	../Macros.h
	../Vector3.h
	../bodyPartRecognizer.cpp
	../bprTree.cpp
	../depthFeature.cpp
	../depthSample.cpp
	../logCat.cpp
)

#openni 需要这个，且需要放在 target_link_libraries 之前：
include_directories($ENV{OPEN_NI_INCLUDE})# ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
link_directories($ENV{OPEN_NI_LIB})# ${OpenCV_LIB_DIR} ${Boost_LIBRARY_DIRS})

add_executable(${_projname} ${${_projname}_SOURCES})
set(OPENNI_LIBRARIES openNI.lib)
target_link_libraries(${_projname} ${OpenCV_LIBS} ${OPENNI_LIBRARIES}) #opencv 居然不需要 include_directories & link_directories



