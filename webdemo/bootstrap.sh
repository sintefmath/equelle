#!/usr/bin/env bash
# Add required repositories
apt-add-repository ppa:opm/ppa
apt-add-repository ppa:ubuntu-toolchain-r/test
apt-get update

# Install web-server components
apt-get install -y nginx nodejs
ln -snf /scripts/nginx-enabled-sites-default /etc/nginx/sites-enabled/default

# Start web server on boot
update-rc.d nginx defaults
service nginx start
service nginx reload 

# Start node server (compiler) on boot
ln -snf /scripts/node-equelle-server.conf /etc/init/node-equelle-server.conf
initctl reload-configuration
start node-equelle-server

# Install required libraries for Equelle
apt-get install -y python-software-properties software-properties-common
apt-get install -y git flex bison g++ make cmake libopm-autodiff libopm-core-dev libopm-autodiff-dev libeigen3-dev libboost1.48-all-dev openmpi-bin libopenmpi-dev gfortran
apt-get -y install gcc-4.7 g++-4.7
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.6 60 --slave /usr/bin/g++ g++ /usr/bin/g++-4.6 
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.7 40 --slave /usr/bin/g++ g++ /usr/bin/g++-4.7 
update-alternatives --set gcc /usr/bin/gcc-4.7

# Clone Equelle git repository, and build
mkdir -p /equelle/src /equelle/build
chown vagrant:users /equelle/src /equelle/build
sudo -u vagrant git clone https://github.com/sintefmath/equelle.git /equelle/src
cd /equelle/build
sudo -u vagrant cmake /equelle/src
sudo -u vagrant make
