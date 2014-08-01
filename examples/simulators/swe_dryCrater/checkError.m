% Compare result from 9th timestep in serial and cuda version

cuda9 = load('q1-00100.output');
serial9 = load('serial_q1-00100.output');

maxDiff = max(abs(cuda9 - serial9))
norm = sqrt(sum(cuda9.*cuda9 - serial9.*serial9))

maxVal = max(cuda9)