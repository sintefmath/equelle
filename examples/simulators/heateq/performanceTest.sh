#!/bin/bash

if [[ -f benchmark.txt ]]; then
    rm benchmark.txt
fi

echo "START MASSIVE TEST" >> benchmark.txt
echo "-------------------" >> benchmark.txt
echo >> benchmark.txt
echo >> benchmark.txt


cases=(1 2 3 4 5)
nx=(50 50  100 100 150)
ny=(50 100 100 100 100)
nz=(50 50  50  100 100)
# (.125 .25 .5 1   1.5)

for i in ${!cases[@]}
do
    echo " " >> benchmark.txt
    echo "--------------------------------------------------" >> benchmark.txt
    echo "Case ${cases[$i]} with cartesian ${nx[$i]} ${ny[$i]} ${nz[$i]}" >> benchmark.txt
    echo "--------------------------------------------------" >> benchmark.txt

    # Change pre-proc script
    sed -i "s/nx =.*/nx = ${nx[$i]};/g" createDirichlet.m
    sed -i "s/ny =.*/ny = ${ny[$i]};/g" createDirichlet.m
    sed -i "s/nz =.*/nz = ${nz[$i]};/g" createDirichlet.m

    octave createDirichlet.m

    sed -i "s/nx=.*/nx=${nx[$i]}/g" params.param
    sed -i "s/ny=.*/ny=${ny[$i]}/g" params.param
    sed -i "s/nz=.*/nz=${nz[$i]}/g" params.param

    for j in 1 2 3
    do
	time optirun ./timer params.param
    done

done
