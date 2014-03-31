#!/usr/bin/env bash

sudo apt-get install -y python-software-properties software-properties-common
sudo apt-add-repository ppa:opm/ppa
sudo apt-get update
sudo apt-get install -y git flex bison g++ make cmake libopm-autodiff libopm-core-dev libopm-autodiff-dev libeigen3-dev libboost-all-dev openmpi-bin libopenmpi-dev gfortran
wget http://trilinos.sandia.gov/download/files/trilinos-11.6.1-Source.tar.bz2
mkdir -p src/build-trilinos
cd src
tar xjf ../trilinos-11.6.1-Source.tar.bz2
cd build-trilinos
cmake ../trilinos-11.6.1-Source/ -DCMAKE_INSTALL_PREFIX=/home/vagrant/hacks -DTrilinos_ENABLE_Zoltan=ON -DTPL_ENABLE_MPI:BOOL=ON -DMPI_BASE_DIR:PATH=/usr/lib/openmpi -DBUILD_SHARED_LIBS:BOOL=ON -DTrilinos_ENABLE_CXX11:BOOL=ON -DCMAKE_CXX_FLAGS:STRING=-std=c++11
make -j2
make install

cd /home/vagrant
mkdir build
cd build
cmake /equelle -DCMAKE_PREFIX_PATH=/home/vagrant/hacks -DEQUELLE_BUILD_MPI=ON
make 
cd /home/vagrant
sudo chown -R vagrant.vagrant build





