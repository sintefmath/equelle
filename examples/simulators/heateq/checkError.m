% comparing results between serial and cuda version

cuda = load('expU-00150.output');
cpu = load('serial_expU-00150.output');

maxCuda = max(cuda)
maxCPU = max(cpu)

maxError = max(abs(cuda - cpu))
norm = sqrt(sum((cuda - cpu).*(cuda - cpu)))