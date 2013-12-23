#!/bin/bash

for i in $(ls dsl/*.equelle)
do
    echo "Compiling $i"
    bn=$(basename $i .equelle)
    ../compiler/ec < $i > "app/out_$bn.cpp"
done
