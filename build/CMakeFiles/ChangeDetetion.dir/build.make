# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

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
CMAKE_SOURCE_DIR = /home/thtf/test/ChangeDetection

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/thtf/test/ChangeDetection/build

# Include any dependencies generated for this target.
include CMakeFiles/ChangeDetetion.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/ChangeDetetion.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ChangeDetetion.dir/flags.make

CMakeFiles/ChangeDetetion.dir/test.cpp.o: CMakeFiles/ChangeDetetion.dir/flags.make
CMakeFiles/ChangeDetetion.dir/test.cpp.o: ../test.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/thtf/test/ChangeDetection/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ChangeDetetion.dir/test.cpp.o"
	/usr/bin/aarch64-linux-gnu-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ChangeDetetion.dir/test.cpp.o -c /home/thtf/test/ChangeDetection/test.cpp

CMakeFiles/ChangeDetetion.dir/test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ChangeDetetion.dir/test.cpp.i"
	/usr/bin/aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/thtf/test/ChangeDetection/test.cpp > CMakeFiles/ChangeDetetion.dir/test.cpp.i

CMakeFiles/ChangeDetetion.dir/test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ChangeDetetion.dir/test.cpp.s"
	/usr/bin/aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/thtf/test/ChangeDetection/test.cpp -o CMakeFiles/ChangeDetetion.dir/test.cpp.s

# Object files for target ChangeDetetion
ChangeDetetion_OBJECTS = \
"CMakeFiles/ChangeDetetion.dir/test.cpp.o"

# External object files for target ChangeDetetion
ChangeDetetion_EXTERNAL_OBJECTS =

ChangeDetetion: CMakeFiles/ChangeDetetion.dir/test.cpp.o
ChangeDetetion: CMakeFiles/ChangeDetetion.dir/build.make
ChangeDetetion: CMakeFiles/ChangeDetetion.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/thtf/test/ChangeDetection/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ChangeDetetion"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ChangeDetetion.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ChangeDetetion.dir/build: ChangeDetetion

.PHONY : CMakeFiles/ChangeDetetion.dir/build

CMakeFiles/ChangeDetetion.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ChangeDetetion.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ChangeDetetion.dir/clean

CMakeFiles/ChangeDetetion.dir/depend:
	cd /home/thtf/test/ChangeDetection/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/thtf/test/ChangeDetection /home/thtf/test/ChangeDetection /home/thtf/test/ChangeDetection/build /home/thtf/test/ChangeDetection/build /home/thtf/test/ChangeDetection/build/CMakeFiles/ChangeDetetion.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ChangeDetetion.dir/depend

