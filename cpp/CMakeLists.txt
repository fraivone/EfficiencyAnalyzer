cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(VFATEfficiencyPlot)

find_package(ROOT)
include(${ROOT_USE_FILE})

include_directories($ENV{ANALYSIS_HOME}/include)
# set(sources $ENV{ANALYSIS_HOME}/src/*.cpp)
set(header $ENV{ANALYSIS_HOME}/include/*.h)

add_executable(VFATEffPlotter VFATEffPlotter.cpp  ${headers})#${sources}
add_executable(Hits_plotter Hits_plotter.cpp  ${headers})#${sources}
target_link_libraries(VFATEffPlotter ${ROOT_LIBRARIES})
target_link_libraries(Hits_plotter ${ROOT_LIBRARIES})

install(TARGETS VFATEffPlotter DESTINATION bin)
install(TARGETS Hits_plotter DESTINATION bin)
