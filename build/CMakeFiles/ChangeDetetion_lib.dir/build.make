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
include CMakeFiles/ChangeDetetion_lib.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/ChangeDetetion_lib.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ChangeDetetion_lib.dir/flags.make

CMakeFiles/ChangeDetetion_lib.dir/src/ChangeDetection.cpp.o: CMakeFiles/ChangeDetetion_lib.dir/flags.make
CMakeFiles/ChangeDetetion_lib.dir/src/ChangeDetection.cpp.o: ../src/ChangeDetection.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/thtf/test/ChangeDetection/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ChangeDetetion_lib.dir/src/ChangeDetection.cpp.o"
	/usr/bin/aarch64-linux-gnu-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ChangeDetetion_lib.dir/src/ChangeDetection.cpp.o -c /home/thtf/test/ChangeDetection/src/ChangeDetection.cpp

CMakeFiles/ChangeDetetion_lib.dir/src/ChangeDetection.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ChangeDetetion_lib.dir/src/ChangeDetection.cpp.i"
	/usr/bin/aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/thtf/test/ChangeDetection/src/ChangeDetection.cpp > CMakeFiles/ChangeDetetion_lib.dir/src/ChangeDetection.cpp.i

CMakeFiles/ChangeDetetion_lib.dir/src/ChangeDetection.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ChangeDetetion_lib.dir/src/ChangeDetection.cpp.s"
	/usr/bin/aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/thtf/test/ChangeDetection/src/ChangeDetection.cpp -o CMakeFiles/ChangeDetetion_lib.dir/src/ChangeDetection.cpp.s

# Object files for target ChangeDetetion_lib
ChangeDetetion_lib_OBJECTS = \
"CMakeFiles/ChangeDetetion_lib.dir/src/ChangeDetection.cpp.o"

# External object files for target ChangeDetetion_lib
ChangeDetetion_lib_EXTERNAL_OBJECTS =

libChangeDetetion_lib.so: CMakeFiles/ChangeDetetion_lib.dir/src/ChangeDetection.cpp.o
libChangeDetetion_lib.so: CMakeFiles/ChangeDetetion_lib.dir/build.make
libChangeDetetion_lib.so: CMakeFiles/ChangeDetetion_lib.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/thtf/test/ChangeDetection/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libChangeDetetion_lib.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ChangeDetetion_lib.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ChangeDetetion_lib.dir/build: libChangeDetetion_lib.so

.PHONY : CMakeFiles/ChangeDetetion_lib.dir/build

CMakeFiles/ChangeDetetion_lib.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ChangeDetetion_lib.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ChangeDetetion_lib.dir/clean

CMakeFiles/ChangeDetetion_lib.dir/depend:
	cd /home/thtf/test/ChangeDetection/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/thtf/test/ChangeDetection /home/thtf/test/ChangeDetection /home/thtf/test/ChangeDetection/build /home/thtf/test/ChangeDetection/build /home/thtf/test/ChangeDetection/build/CMakeFiles/ChangeDetetion_lib.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ChangeDetetion_lib.dir/depend
