# Import necessary packages
using LinearAlgebra
#using BLIS  # BLIS.jl for using BLIS directly

# Set the number of BLAS threads
blas_threads = 1
BLAS.set_num_threads(blas_threads)
println("Number of BLAS threads set to: $blas_threads")

# Example test function to benchmark linear algebra operations
function test_linear_algebra()
    A = rand(1000, 1000)
    B = rand(1000, 1000)
    C = A * B
    #println("Matrix multiplication completed.")
end

# Benchmarking example
using BenchmarkTools
println("Benchmarking matrix multiplication with BLIS:")
@benchmark test_linear_algebra()