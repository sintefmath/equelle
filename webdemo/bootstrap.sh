#!/usr/bin/env bash

# Update distribution
apt-get update
apt-get -y dist-upgrade

# Add required repositories
apt-get install -y python-software-properties software-properties-common
apt-add-repository --yes ppa:opm/ppa
apt-add-repository --yes ppa:ubuntu-toolchain-r/test
apt-add-repository --yes ppa:chris-lea/node.js
apt-add-repository --yes ppa:nginx/stable
apt-get update

# Install required libraries
apt-get install -y git flex bison g++ make cmake libopm-autodiff libopm-core-dev libopm-autodiff-dev libeigen3-dev libboost-all-dev openmpi-bin libopenmpi-dev gfortran
apt-get -y install gcc-4.7 g++-4.7
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.6 60 --slave /usr/bin/g++ g++ /usr/bin/g++-4.6
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.7 40 --slave /usr/bin/g++ g++ /usr/bin/g++-4.7
update-alternatives --set gcc /usr/bin/gcc-4.7

# Clone Git repository
mkdir -p /equelle/src /equelle/build
useradd -G users equelle
chown equelle:users /equelle/src /equelle/build

sudo -u equelle git clone https://github.com/sintefmath/equelle.git /equelle/src
cd /equelle/src
sudo -u equelle git submodule update --init --recursive

# Install web server components
apt-get install -y nginx nodejs
ln -snf /scripts/nginx-enabled-sites-default /etc/nginx/sites-enabled/default

# Start web server on boot
update-rc.d nginx defaults
service nginx start

# Generate a random secret key for node server to use when signing responses
if [ -s /srv/server/secretkey ]; then
	echo "Secret key already generated."
else
	openssl rand -out /srv/server/secretkey -hex 64
fi

# Start node server (compiler) on boot
ln -snf /scripts/node-equelle-server.conf /etc/init/node-equelle-server.conf
initctl reload-configuration
start node-equelle-server

sh update.sh

# Print complete messages
echo "The Equelle kitchen sink is started. Visit: http://localhost:8080/ to try it out."
echo "To view log files, ssh into the server using 'vagrant ssh', then view the file '/var/log/upstart/node-equelle-server.log'"
