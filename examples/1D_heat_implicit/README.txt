In order to build these test programs on an Ubuntu 12.10-machine, I performed
the following steps. It may be possible for you to simply copy and paste these
commands into a shell (I use tcsh) to obtain the same setup, but your mileage
may vary. In particular, your current set of installed packages may not be the
same as what I started out with.

Note that building opm-autodiff failed with several gcc-versions, e.g., 4.7.2
and 4.6.3. I was successful with version 4.5.4.

When these packages are in place, it should be possible to build the test
programs with the help of the CMakeLists.txt. If the current directory is the
one containing the CMakeLists.txt, one may do e.g. this

  mkdir build
  cd build
  cmake ..
  make -j 4

J.O.



================================================================================



sudo apt-get install libdune-common-dev libdune-istl-dev libumfpack5.4.0 libsuitesparse-dev libsuperlu3-dev
sudo ldconfig

cd
mkdir -p new_system/prosjekter

cd ~/new_system/prosjekter
mkdir -p opm-core
cd opm-core
rm -rf opm-core
git clone https://github.com/OPM/opm-core.git
cd opm-core
mkdir -p build
cd build
cmake ..

make -j 8

cd ~/new_system/prosjekter
mkdir -p opm_ad
cd opm_ad
rm -rf opm-autodiff
git clone https://github.com/OPM/opm-autodiff.git
cd opm-autodiff/

mkdir -p build
cd build
cmake -Dopm-core_INCLUDE_DIR=${HOME}/new_system/prosjekter/opm-core/opm-core -Dopm-core_LIBRARY=${HOME}/new_system/prosjekter/opm-core/opm-core/build/lib/libopmcore.a ..

make -j 8
