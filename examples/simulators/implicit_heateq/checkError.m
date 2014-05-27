% comparing results between serial and cuda version

cuda = load('u-00020.output');
cpu = load('cpu_u-00020.output');

maxCuda = max(cuda)
maxCPU = max(cpu)

maxError = max(abs(cuda - cpu))
norm = sqrt(sum((cuda - cpu).*(cuda - cpu)))