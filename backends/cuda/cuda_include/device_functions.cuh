#ifndef EQUELLE_DEVICE_FUNCTIONS_CUH_INCLUDED
#define EQUELLE_DEVICE_FUNCTIONS_CUH_INCLUDED

// Implement this once, and only once in order to avoid 
// typos that might cause bugs.
// Need to enfore inlining to avoid multiple definitions
// see e.g: http://choorucode.com/2011/03/15/cuda-device-function-in-header-file/
__forceinline__ __device__ int myID(){
    return (threadIdx.x + blockIdx.x*blockDim.x);
}

#endif // EQUELLE_DEVICE_FUNCTIONS_CUH_INCLUDED