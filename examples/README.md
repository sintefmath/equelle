Equelle_examples
==============

This directory contains various example and benchmark Equelle programs. It is designed to be compiled as a standalone project, and the recommended way of building the examples is in form of an 
out-of-source build.

Assume you have the Equelle sources in $EQUELLE_SRC and you have built Equelle in $EQUELLE_BUILD, then the following command line should set you up nicely:

``` 
$ mkdir equelle-examples-build
$ cd equelle-examples-build
$ cmake $EQUELLE_SRC/examples -DCMAKE_PREFIX_PATH=$EQUELLE_BUILD
```

You might also optionally add -DEQUELLE_USE_MPI=ON if you want to build MPI examples.

