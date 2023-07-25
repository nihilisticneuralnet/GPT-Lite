import Pkg
Pkg.add("CUDA")

using CUDA
if CUDA.functional()
    println("CUDA is functional.")
    device_count = CUDA.device_count()
    println("Number of CUDA devices: $device_count")
else
    println("CUDA is not functional. Please check your installation.")
end


using Flux
using Flux: @epochs, onehotbatch, mse, throttle
using CUDA

# hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = cpu() 
