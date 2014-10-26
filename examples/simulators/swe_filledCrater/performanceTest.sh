#!/bin/bash

# SHALLOW WATER EQUATIONS

if [[ -f benchmark.txt ]]; then
    rm benchmark.txt
fi

echo "START MASSIVE TEST" >> benchmark.txt
echo "-------------------" >> benchmark.txt
echo >> benchmark.txt
echo >> benchmark.txt


cases=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
nx=(50  100 200 300 200 200 200 200 200 300 500 600 300 500 500 500 1000 1000 1000 1500)
ny=(100 100 100 100 200 250 300 400 500 400 300 300 700 500 750 1000 750 1000 1250 1000)
cpu=(0 1 0 1 1 1 1 1 1 1 0 0 0 0 1 0 1 0 0 0) #bool 
#nz=(50 50  50  100 100 100 100 100 100 100)
# (.125 .25 .5 1   1.5 2.0 2.5 3.0 3.5 3.75)

for i in ${!cases[@]}
do
    echo " " >> benchmark.txt
    echo "--------------------------------------------------" >> benchmark.txt
    echo "Case ${cases[$i]} with cartesian ${nx[$i]} ${ny[$i]}" >> benchmark.txt
    echo "--------------------------------------------------" >> benchmark.txt

    # Change pre-proc script
    echo "1"
    sed -i "s/nx =.*/nx = ${nx[$i]};/g" constructCase.m
    echo "2"
    sed -i "s/ny =.*/ny = ${ny[$i]};/g" constructCase.m
    echo "3"
    sed -i "s/timesteps =.*/timesteps = 120;/g" constructCase.m

    octave constructCase.m
    
    echo "4"
    sed -i "s/nx=.*/nx=${nx[$i]}/g" case.param
    echo "5"
    sed -i "s/ny=.*/ny=${ny[$i]}/g" case.param


    for j in 1 2 3
    do
	time optirun ./timer case.param
    done
    echo "And serial:" >> benchmark.txt
    if [[ "${cpu[$i]}" -eq 1 ]]
    then
	#time ./serial_timer case.param
	echo "Omit serial" >> benchmark.txt
	echo "Omit serial"
    else
	echo "Omit serial" >> benchmark.txt
	echo "Omit serial"
    fi

done
