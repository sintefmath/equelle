#!/bin/bash

for i in $(ls dsl/*.equelle)
do
    echo "Compiling $i"
    bn=$(basename $i .equelle)
    ../../equelle-build/compiler/ec --nondimensional -i $i --backend=cpu > "app/out_$bn.cpp"
done
