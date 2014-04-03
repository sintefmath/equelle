Vagrant
=======

[Vagrant](www.vagrantup.com) is a tool for creating and configuring reproducible and
portable development environments. The files in this directory is
all that is needed to bring up (including downloading the base image)
an Ubuntu 13.10 environment that can be used to compile and run the serial
and MPI backends.

It requires [VirtualBox](www.virtualbox.org) and Vagrant to be installed.
They are apt-gettable on recent Ubuntus and have relatively hassle-free installers
on Windows and OSX.


To use vagrant execute the following
```
equelle/vagrant$ vagrant up
```

The first time this is executed will take around 10 minutes, as the image
must be downloaded, OPM will be installed and Zoltan will be compiled.

If executed from the checked out repository, the virtual machine will have
the repo (from the host, not github) mounted at /equelle.

Furthermore it will configure and build Equelle in /home/vagrant/build.

Now execute the following to log into the virtual machine as the vagrant user
```
equelle/vagrant$ vagrant ssh
```

From here you can build and execute tests (also with mpirun) in a reproducible environment.
