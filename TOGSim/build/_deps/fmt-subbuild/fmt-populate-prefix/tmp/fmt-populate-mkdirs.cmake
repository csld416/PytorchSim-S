# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/workspace/legomerged/eclab_legosim/PyTorchSim/TOGSim/extern/ramulator2/ext/fmt"
  "/workspace/legomerged/eclab_legosim/PyTorchSim/TOGSim/build/_deps/fmt-build"
  "/workspace/legomerged/eclab_legosim/PyTorchSim/TOGSim/build/_deps/fmt-subbuild/fmt-populate-prefix"
  "/workspace/legomerged/eclab_legosim/PyTorchSim/TOGSim/build/_deps/fmt-subbuild/fmt-populate-prefix/tmp"
  "/workspace/legomerged/eclab_legosim/PyTorchSim/TOGSim/build/_deps/fmt-subbuild/fmt-populate-prefix/src/fmt-populate-stamp"
  "/workspace/legomerged/eclab_legosim/PyTorchSim/TOGSim/build/_deps/fmt-subbuild/fmt-populate-prefix/src"
  "/workspace/legomerged/eclab_legosim/PyTorchSim/TOGSim/build/_deps/fmt-subbuild/fmt-populate-prefix/src/fmt-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/workspace/legomerged/eclab_legosim/PyTorchSim/TOGSim/build/_deps/fmt-subbuild/fmt-populate-prefix/src/fmt-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/workspace/legomerged/eclab_legosim/PyTorchSim/TOGSim/build/_deps/fmt-subbuild/fmt-populate-prefix/src/fmt-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
