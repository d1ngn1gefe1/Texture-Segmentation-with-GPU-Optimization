#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//Declare the functions that will be used in the kernel...
cudaError_t addWithCuda(int *c, const int *a, const int *b, size_t size);
cudaError_t addWithCuda(int *c, const int *a, const int *b, size_t size);