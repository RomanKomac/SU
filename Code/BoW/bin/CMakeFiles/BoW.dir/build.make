# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.9

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/roman/SU/Code/BoW

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/roman/SU/Code/BoW/bin

# Include any dependencies generated for this target.
include CMakeFiles/BoW.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/BoW.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/BoW.dir/flags.make

CMakeFiles/BoW.dir/home/roman/SU/Code/Common/Image.cpp.o: CMakeFiles/BoW.dir/flags.make
CMakeFiles/BoW.dir/home/roman/SU/Code/Common/Image.cpp.o: /home/roman/SU/Code/Common/Image.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/roman/SU/Code/BoW/bin/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/BoW.dir/home/roman/SU/Code/Common/Image.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/BoW.dir/home/roman/SU/Code/Common/Image.cpp.o -c /home/roman/SU/Code/Common/Image.cpp

CMakeFiles/BoW.dir/home/roman/SU/Code/Common/Image.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/BoW.dir/home/roman/SU/Code/Common/Image.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/roman/SU/Code/Common/Image.cpp > CMakeFiles/BoW.dir/home/roman/SU/Code/Common/Image.cpp.i

CMakeFiles/BoW.dir/home/roman/SU/Code/Common/Image.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/BoW.dir/home/roman/SU/Code/Common/Image.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/roman/SU/Code/Common/Image.cpp -o CMakeFiles/BoW.dir/home/roman/SU/Code/Common/Image.cpp.s

CMakeFiles/BoW.dir/home/roman/SU/Code/Common/Image.cpp.o.requires:

.PHONY : CMakeFiles/BoW.dir/home/roman/SU/Code/Common/Image.cpp.o.requires

CMakeFiles/BoW.dir/home/roman/SU/Code/Common/Image.cpp.o.provides: CMakeFiles/BoW.dir/home/roman/SU/Code/Common/Image.cpp.o.requires
	$(MAKE) -f CMakeFiles/BoW.dir/build.make CMakeFiles/BoW.dir/home/roman/SU/Code/Common/Image.cpp.o.provides.build
.PHONY : CMakeFiles/BoW.dir/home/roman/SU/Code/Common/Image.cpp.o.provides

CMakeFiles/BoW.dir/home/roman/SU/Code/Common/Image.cpp.o.provides.build: CMakeFiles/BoW.dir/home/roman/SU/Code/Common/Image.cpp.o


CMakeFiles/BoW.dir/Learning/learning.cpp.o: CMakeFiles/BoW.dir/flags.make
CMakeFiles/BoW.dir/Learning/learning.cpp.o: ../Learning/learning.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/roman/SU/Code/BoW/bin/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/BoW.dir/Learning/learning.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/BoW.dir/Learning/learning.cpp.o -c /home/roman/SU/Code/BoW/Learning/learning.cpp

CMakeFiles/BoW.dir/Learning/learning.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/BoW.dir/Learning/learning.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/roman/SU/Code/BoW/Learning/learning.cpp > CMakeFiles/BoW.dir/Learning/learning.cpp.i

CMakeFiles/BoW.dir/Learning/learning.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/BoW.dir/Learning/learning.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/roman/SU/Code/BoW/Learning/learning.cpp -o CMakeFiles/BoW.dir/Learning/learning.cpp.s

CMakeFiles/BoW.dir/Learning/learning.cpp.o.requires:

.PHONY : CMakeFiles/BoW.dir/Learning/learning.cpp.o.requires

CMakeFiles/BoW.dir/Learning/learning.cpp.o.provides: CMakeFiles/BoW.dir/Learning/learning.cpp.o.requires
	$(MAKE) -f CMakeFiles/BoW.dir/build.make CMakeFiles/BoW.dir/Learning/learning.cpp.o.provides.build
.PHONY : CMakeFiles/BoW.dir/Learning/learning.cpp.o.provides

CMakeFiles/BoW.dir/Learning/learning.cpp.o.provides.build: CMakeFiles/BoW.dir/Learning/learning.cpp.o


CMakeFiles/BoW.dir/bow_main.cpp.o: CMakeFiles/BoW.dir/flags.make
CMakeFiles/BoW.dir/bow_main.cpp.o: ../bow_main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/roman/SU/Code/BoW/bin/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/BoW.dir/bow_main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/BoW.dir/bow_main.cpp.o -c /home/roman/SU/Code/BoW/bow_main.cpp

CMakeFiles/BoW.dir/bow_main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/BoW.dir/bow_main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/roman/SU/Code/BoW/bow_main.cpp > CMakeFiles/BoW.dir/bow_main.cpp.i

CMakeFiles/BoW.dir/bow_main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/BoW.dir/bow_main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/roman/SU/Code/BoW/bow_main.cpp -o CMakeFiles/BoW.dir/bow_main.cpp.s

CMakeFiles/BoW.dir/bow_main.cpp.o.requires:

.PHONY : CMakeFiles/BoW.dir/bow_main.cpp.o.requires

CMakeFiles/BoW.dir/bow_main.cpp.o.provides: CMakeFiles/BoW.dir/bow_main.cpp.o.requires
	$(MAKE) -f CMakeFiles/BoW.dir/build.make CMakeFiles/BoW.dir/bow_main.cpp.o.provides.build
.PHONY : CMakeFiles/BoW.dir/bow_main.cpp.o.provides

CMakeFiles/BoW.dir/bow_main.cpp.o.provides.build: CMakeFiles/BoW.dir/bow_main.cpp.o


# Object files for target BoW
BoW_OBJECTS = \
"CMakeFiles/BoW.dir/home/roman/SU/Code/Common/Image.cpp.o" \
"CMakeFiles/BoW.dir/Learning/learning.cpp.o" \
"CMakeFiles/BoW.dir/bow_main.cpp.o"

# External object files for target BoW
BoW_EXTERNAL_OBJECTS =

BoW: CMakeFiles/BoW.dir/home/roman/SU/Code/Common/Image.cpp.o
BoW: CMakeFiles/BoW.dir/Learning/learning.cpp.o
BoW: CMakeFiles/BoW.dir/bow_main.cpp.o
BoW: CMakeFiles/BoW.dir/build.make
BoW: /usr/local/lib/libopencv_stitching.so.3.3.1
BoW: /usr/local/lib/libopencv_superres.so.3.3.1
BoW: /usr/local/lib/libopencv_videostab.so.3.3.1
BoW: /usr/local/lib/libopencv_aruco.so.3.3.1
BoW: /usr/local/lib/libopencv_bgsegm.so.3.3.1
BoW: /usr/local/lib/libopencv_bioinspired.so.3.3.1
BoW: /usr/local/lib/libopencv_ccalib.so.3.3.1
BoW: /usr/local/lib/libopencv_dpm.so.3.3.1
BoW: /usr/local/lib/libopencv_face.so.3.3.1
BoW: /usr/local/lib/libopencv_freetype.so.3.3.1
BoW: /usr/local/lib/libopencv_fuzzy.so.3.3.1
BoW: /usr/local/lib/libopencv_hdf.so.3.3.1
BoW: /usr/local/lib/libopencv_img_hash.so.3.3.1
BoW: /usr/local/lib/libopencv_line_descriptor.so.3.3.1
BoW: /usr/local/lib/libopencv_optflow.so.3.3.1
BoW: /usr/local/lib/libopencv_reg.so.3.3.1
BoW: /usr/local/lib/libopencv_rgbd.so.3.3.1
BoW: /usr/local/lib/libopencv_stereo.so.3.3.1
BoW: /usr/local/lib/libopencv_structured_light.so.3.3.1
BoW: /usr/local/lib/libopencv_surface_matching.so.3.3.1
BoW: /usr/local/lib/libopencv_tracking.so.3.3.1
BoW: /usr/local/lib/libopencv_xfeatures2d.so.3.3.1
BoW: /usr/local/lib/libopencv_ximgproc.so.3.3.1
BoW: /usr/local/lib/libopencv_xobjdetect.so.3.3.1
BoW: /usr/local/lib/libopencv_xphoto.so.3.3.1
BoW: /usr/local/lib/libopencv_shape.so.3.3.1
BoW: /usr/local/lib/libopencv_photo.so.3.3.1
BoW: /usr/local/lib/libopencv_calib3d.so.3.3.1
BoW: /usr/local/lib/libopencv_viz.so.3.3.1
BoW: /usr/local/lib/libopencv_phase_unwrapping.so.3.3.1
BoW: /usr/local/lib/libopencv_video.so.3.3.1
BoW: /usr/local/lib/libopencv_datasets.so.3.3.1
BoW: /usr/local/lib/libopencv_plot.so.3.3.1
BoW: /usr/local/lib/libopencv_text.so.3.3.1
BoW: /usr/local/lib/libopencv_dnn.so.3.3.1
BoW: /usr/local/lib/libopencv_features2d.so.3.3.1
BoW: /usr/local/lib/libopencv_flann.so.3.3.1
BoW: /usr/local/lib/libopencv_highgui.so.3.3.1
BoW: /usr/local/lib/libopencv_ml.so.3.3.1
BoW: /usr/local/lib/libopencv_videoio.so.3.3.1
BoW: /usr/local/lib/libopencv_imgcodecs.so.3.3.1
BoW: /usr/local/lib/libopencv_objdetect.so.3.3.1
BoW: /usr/local/lib/libopencv_imgproc.so.3.3.1
BoW: /usr/local/lib/libopencv_core.so.3.3.1
BoW: CMakeFiles/BoW.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/roman/SU/Code/BoW/bin/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable BoW"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/BoW.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/BoW.dir/build: BoW

.PHONY : CMakeFiles/BoW.dir/build

CMakeFiles/BoW.dir/requires: CMakeFiles/BoW.dir/home/roman/SU/Code/Common/Image.cpp.o.requires
CMakeFiles/BoW.dir/requires: CMakeFiles/BoW.dir/Learning/learning.cpp.o.requires
CMakeFiles/BoW.dir/requires: CMakeFiles/BoW.dir/bow_main.cpp.o.requires

.PHONY : CMakeFiles/BoW.dir/requires

CMakeFiles/BoW.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/BoW.dir/cmake_clean.cmake
.PHONY : CMakeFiles/BoW.dir/clean

CMakeFiles/BoW.dir/depend:
	cd /home/roman/SU/Code/BoW/bin && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/roman/SU/Code/BoW /home/roman/SU/Code/BoW /home/roman/SU/Code/BoW/bin /home/roman/SU/Code/BoW/bin /home/roman/SU/Code/BoW/bin/CMakeFiles/BoW.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/BoW.dir/depend

