# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/Users/swk/Documents/swk/could delete/aligator/dual-FFSR_ProxDDP/_deps/jrl-cmakemodules-src")
  file(MAKE_DIRECTORY "/Users/swk/Documents/swk/could delete/aligator/dual-FFSR_ProxDDP/_deps/jrl-cmakemodules-src")
endif()
file(MAKE_DIRECTORY
  "/Users/swk/Documents/swk/could delete/aligator/dual-FFSR_ProxDDP/_deps/jrl-cmakemodules-build"
  "/Users/swk/Documents/swk/could delete/aligator/dual-FFSR_ProxDDP/_deps/jrl-cmakemodules-subbuild/jrl-cmakemodules-populate-prefix"
  "/Users/swk/Documents/swk/could delete/aligator/dual-FFSR_ProxDDP/_deps/jrl-cmakemodules-subbuild/jrl-cmakemodules-populate-prefix/tmp"
  "/Users/swk/Documents/swk/could delete/aligator/dual-FFSR_ProxDDP/_deps/jrl-cmakemodules-subbuild/jrl-cmakemodules-populate-prefix/src/jrl-cmakemodules-populate-stamp"
  "/Users/swk/Documents/swk/could delete/aligator/dual-FFSR_ProxDDP/_deps/jrl-cmakemodules-subbuild/jrl-cmakemodules-populate-prefix/src"
  "/Users/swk/Documents/swk/could delete/aligator/dual-FFSR_ProxDDP/_deps/jrl-cmakemodules-subbuild/jrl-cmakemodules-populate-prefix/src/jrl-cmakemodules-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/Users/swk/Documents/swk/could delete/aligator/dual-FFSR_ProxDDP/_deps/jrl-cmakemodules-subbuild/jrl-cmakemodules-populate-prefix/src/jrl-cmakemodules-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/Users/swk/Documents/swk/could delete/aligator/dual-FFSR_ProxDDP/_deps/jrl-cmakemodules-subbuild/jrl-cmakemodules-populate-prefix/src/jrl-cmakemodules-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
