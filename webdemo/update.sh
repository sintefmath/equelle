#!/usr/bin/env bash

# Update distribution
apt-get update
apt-get -y dist-upgrade

# Pull from Git repository and build
cd /equelle/src
sudo -u equelle git pull origin master
sudo -u equelle git submodule update --init --recursive

cd /equelle/build
sudo -u equelle cmake /equelle/src
sudo -u equelle make

rm -rf /srv && ln -snf /equelle/src/webdemo/srv /srv
rm -rf /scripts && ln -snf /equelle/src/webdemo/scripts /scripts

# Copy examples to web server
sudo cp /equelle/src/examples/dsl/twophase.equelle /srv/examples/3dwell/flow.equelle
sudo cp /equelle/src/examples/dsl/heateq.equelle /srv/examples/2dheateq/heateq.equelle

# Restart web server
service nginx reload

# Restart node server (compiler)
restart node-equelle-server

# Generate syntax highlighting and code completion scripts
cd /scripts/codemirror_mode_generator
sudo -u equelle ./generate.py > /srv/client/js/equelleMode.js
sudo -u equelle ./generateHints.py > /srv/client/js/equelleHints.js

# Initialize XTK forked repository
cd /srv/XTK/utils
./deps.py
