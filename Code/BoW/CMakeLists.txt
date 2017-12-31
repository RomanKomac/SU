cmake_minimum_required(VERSION 2.8)

#--------------------------------------------------------------------
#Build definitions
#--------------------------------------------------------------------
#Required cpp version 
macro(use_cxx11)
  if (CMAKE_VERSION VERSION_LESS "3.1")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      set (CMAKE_CXX_FLAGS "-std=gnu++11 ${CMAKE_CXX_FLAGS}")
    endif ()
  else ()
    set (CMAKE_CXX_STANDARD 11)
  endif ()
endmacro(use_cxx11)

#Project name
project( BagOfWords_CV )

#Build mode (DEBUG, VERBOSE, RELEASE)
add_definitions(-D VERBOSE)


#--------------------------------------------------------------------
#Package definitions
#--------------------------------------------------------------------
#OpenCV library
find_package( OpenCV REQUIRED )

#--------------------------------------------------------------------
#File definitions
#--------------------------------------------------------------------
#Testing files
file(GLOB ESTIMATOR_FILES
    "Testing/*.hpp"
    "Testing/*.cpp"
)

#Learning files
file(GLOB LEARNING_FILES
    "Learning/*.hpp"
    "Learning/*.cpp"
)

#Helper files, such as loaders, readers, writers
set( HELPER_FILES ../Common/Image.hpp ../Common/Image.cpp )

#Predefined constants and definitions
set( PREDEF_FILES ../Common/const_defs.hpp ../Common/typedefs.hpp )


add_executable( BoW ${HELPER_FILES} ${TESTING_FILES} ${LEARNING_FILES} ${PREDEF_FILES} bow_main.cpp )
target_link_libraries( BoW ${OpenCV_LIBS})
