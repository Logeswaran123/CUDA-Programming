# CUDA Programming
My CUDA Programming learning journey 🐜🐜🐜


## Basics
The basic flow in CUDA programming is,
1. Initialize data from CPU.
2. Transfer data from CPU to GPU.
3. Lauch necessary kernel executions on data.
4. Tranfer data from GPU to CPU.
5. Release memory from CPU and GPU.

### Terms usage
Host - CPU <br />
Device - GPU

Essentially, Host runs sequential operations and Device runs parallel operations.

General syntax for launching a kernel, <br />
```
kernel_name<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(arguments);
```
