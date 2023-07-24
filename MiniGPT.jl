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
