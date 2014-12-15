#!/bin/bash

for i in $(ls dsl/*.equelle)
do
    echo "Compiling $i"
    bn=$(basename $i .equelle)
    nondim="swe twophase_fully_implicit twophase_fully_implicit_conservation twophase_grav"
    if [[ $nondim =~ (^| )$bn($| ) ]]; then
        ../../equelle-build/compiler/ec -i $i --nondimensional --backend=cpu > "app/out_$bn.cpp"
    else
        ../../equelle-build/compiler/ec -i $i --backend=cpu > "app/out_$bn.cpp"
    fi
done
