#!/usr/bin/env bash
# Add required repositories
apt-get update
apt-get -y dist-upgrade
apt-get install -y python-software-properties software-properties-common
apt-add-repository --yes ppa:opm/ppa
apt-add-repository --yes ppa:ubuntu-toolchain-r/test
apt-add-repository --yes ppa:chris-lea/node.js
apt-add-repository --yes ppa:nginx/stable
apt-get update

# Install required libraries for Equelle
apt-get install -y git flex bison g++ make cmake libopm-autodiff libopm-core-dev libopm-autodiff-dev libeigen3-dev libboost-all-dev openmpi-bin libopenmpi-dev gfortran
apt-get -y install gcc-4.7 g++-4.7
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.6 60 --slave /usr/bin/g++ g++ /usr/bin/g++-4.6
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.7 40 --slave /usr/bin/g++ g++ /usr/bin/g++-4.7
update-alternatives --set gcc /usr/bin/gcc-4.7

# Clone Equelle git repository, and build
# TODO: Change to sintefmath repository.
mkdir -p /equelle/src /equelle/build
useradd -G users equelle
chown equelle:users /equelle/src /equelle/build
sudo -u equelle git clone https://github.com/sintefmath/equelle.git /equelle/src
cd /equelle/src
#sudo -u equelle git reset --hard 659d849c0a59dd33146bf7118d299911787a1cc8 #Just after merge of Jakhog
#sudo -u equelle git reset --hard dcf3c82b87c2cdc956027b64a3f3bb833b98c8c1
sudo -u equelle git submodule update --init --recursive
cd /equelle/build
sudo -u equelle cmake /equelle/src
sudo -u equelle make
rm -rf /srv
ln -snf /equelle/src/webdemo/srv /srv
ln -snf /equelle/src/webdemo/scripts /scripts

#Copy examples to examples dir
sudo cp /equelle/src/examples/dsl/twophase.equelle /srv/examples/3dwell/flow.equelle 
sudo cp /equelle/src/examples/dsl/heateq_timesteps.equelle /srv/examples/2dheateq/heateq.equelle 

# Install web-server components
apt-get install -y nginx nodejs
ln -snf /scripts/nginx-enabled-sites-default /etc/nginx/sites-enabled/default

# Start web server on boot
update-rc.d nginx defaults
service nginx start
service nginx reload

# Generate a random secret key for node to use when signing responses
if [ -s /srv/server/secretkey ]; then
	echo "Secret key already generated."
else
	openssl rand -out /srv/server/secretkey -hex 64
fi

# Start node server (compiler) on boot
ln -snf /scripts/node-equelle-server.conf /etc/init/node-equelle-server.conf
initctl reload-configuration
start node-equelle-server

# Generate Equelle syntax highlighting and code-completion scripts
cd /scripts/codemirror_mode_generator
sudo -u equelle ./generate.py > /srv/client/js/equelleMode.js
sudo -u equelle ./generateHints.py > /srv/client/js/equelleHints.js

# Initialize XTK forked repository
cd /srv/XTK/utils
./deps.py

# Print complete messages
echo "The Equelle kitchen sink is started. Visit: http://localhost:8080/ to try it out."
echo "To view log files, ssh into the server using 'vagrant ssh', then view the file '/var/log/upstart/node-equelle-server.log'"
