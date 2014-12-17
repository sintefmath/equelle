This folder contains matlab/octave scripts for generating input for the Shallow Water simulator swe.equelle, as well as processing and visualize the result.

Run as follows:

1. Build the Equelle compiler in a folder <build_base>.

2. Compile the Equelle shallow water simulator inside this folder:
```$ <build_base>/compiler/ec -i ../../dsl/swe.equelle > swe.cpp```

3. Generate simulator input, by running shore.m in MATLAB or Octave. For example:
```$ octave shore.m```
This should generate a bottom topography formed as a crater on a slope, and a list of timesteps.
It should also generate initial water state, where the body of water is placed in the uppermost corner of the domain with zero initial velocity. Feel free to do changes in this file, and specially the number of timesteps can easily be edited here.

4. Create a new build directory inside this folder (this will be assumed by the parameter file):
```$ mkdir build; cd build```

5. Compile the Equelle generated C++ program using cmake:
```$ cmake ../ -DEquelle_DIR=<build_base>; make```

6. Run the simulator from the new build directory:
```$ ./swe ../shallowWater.param```
If all steps have been successful, the build folder should now be filled with files with extension output.

7. Do post processing of the result. This step requires MATLAB to be run as is, as the light configurations in the surf-plot is not supported by Octave. Feel therefore free to play around with these settings. To visualize the result as is, open Matlab and run 
```outputDir='build'; postProc2D```

