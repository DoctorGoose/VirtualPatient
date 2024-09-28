using Distributed
using LinearAlgebra

blas_threads = 1
procs = 1

BLAS.set_num_threads(blas_threads)

if nprocs() > procs
    rmprocs(workers())
    addprocs(procs - 1) 
elseif nprocs() < procs
    addprocs(procs - nprocs())
end

@everywhere using LinearAlgebra
@everywhere using Random
@everywhere using Statistics
@everywhere using DifferentialEquations
@everywhere using Logging
@everywhere using Profile
@everywhere using SparseArrays
@everywhere using Sundials
@everywhere using OrdinaryDiffEq
#@everywhere using StaticArrays

println("Number of BLAS threads: ", BLAS.get_num_threads())
println("Number of processes: ", procs)
println("Number of threads: ", Threads.nthreads())

@everywhere function suppress_warnings(f)
    logger = ConsoleLogger(stderr, Logging.Error)  # Only show errors and above
    return with_logger(logger) do
        f()
    end
end

@everywhere function LCTModel!(du, u, p, t)
    @inbounds begin
        #u = @SVector u #A lot to implement, uncertain gains
        #du = @MVector du
        # Unpack states
        T, I1, I2, V, E, El = u[1:6]
        z = u[7:end]

        # Unpack parameters
        beta, k, delta, delta_E, K_delta_E, p_param, c, xi, tau, a, d_E = p

        # Precompute constants
        tau_inv = Float32(1.0) / tau
        xi_tau_inv = xi * tau_inv
        delta_E_term = delta_E * E * I2 / (K_delta_E + I2)

        # ODEs
        du[1] = -beta * T * V
        du[2] = beta * T * V - k * I1
        du[3] = k * I1 - delta * I2 - delta_E_term
        du[4] = p_param * I2 - c * V
        du[5] = xi_tau_inv * z[end] - d_E * E
        du[6] = d_E * E  # Lung T cells

        # Delayed variables
        du[7] = tau_inv * (a * I2 - z[1])  # dz_dt[1]
        du[8:end] .= tau_inv .* (z[1:end-1] .- z[2:end])  # dz_dt[2:end]
    end
    return nothing
end

@everywhere function solve_LCTModel(tspan, y0, params)

    y0_float64 = Float64.(collect(y0))
    params_float64 = Float64.(params)
    tspan_float64 = Float64.(tspan)
    solver = CVODE_BDF(
        method = :Newton,
        linear_solver = :GMRES,
        # krylov_dim = 5,            
        # max_nonlinear_iters = 3,   
        # max_convergence_failures = 10,
    )
    prob = ODEProblem(LCTModel!, y0_float64, tspan_float64, params_float64)
    sol = suppress_warnings(() ->solve(prob, solver; reltol=1e-5, abstol=1e-4))


    #prob = ODEProblem(LCTModel!, y0, tspan, params)
    #sol = suppress_warnings(() -> solve(prob, Rodas5(), reltol = 1e-7, abstol = 1e-4)) #TRBDF2(), Rodas5() autodiff = true
    return sol
end

# Function to sample parameters uniformly in log space
@everywhere function sample_parameters(seed, n_samples)
    Random.seed!(seed)
    rng = Random.GLOBAL_RNG
    xi_range = (Float32(1e-2), Float32(1.0))
    a_range = (Float32(1e-1), Float32(1.0))
    tau_range = (Float32(1e-1), Float32(5.0))
    d_E_range = (Float32(0.5), Float32(10.0))
    delta_E_range = (Float32(1e-1), Float32(1e2))
    K_delta_E_range = (Float32(1e2), Float32(1e6))

    param_samples = [(
        exp(rand(rng, Float32) * (log(xi_range[2]) - log(xi_range[1])) + log(xi_range[1])),
        exp(rand(rng, Float32) * (log(a_range[2]) - log(a_range[1])) + log(a_range[1])),
        exp(rand(rng, Float32) * (log(tau_range[2]) - log(tau_range[1])) + log(tau_range[1])),
        exp(rand(rng, Float32) * (log(d_E_range[2]) - log(d_E_range[1])) + log(d_E_range[1])),
        exp(rand(rng, Float32) * (log(delta_E_range[2]) - log(delta_E_range[1])) + log(delta_E_range[1])),
        exp(rand(rng, Float32) * (log(K_delta_E_range[2]) - log(K_delta_E_range[1])) + log(K_delta_E_range[1]))
    ) for _ in 1:n_samples]
    return param_samples
end

# Combine fixed parameters with sampled parameters
@everywhere function create_full_parameter_sets(sampled_params)
    fixed_params = (
        beta=Float32(1.0888e-4),
        k=Float32(4.0),
        delta=Float32(0.96),
        p=Float32(0.02978),
        c=Float32(13.934)
    )
    return [Float32[
        fixed_params.beta,
        fixed_params.k,
        fixed_params.delta,
        delta_E,
        K_delta_E,
        fixed_params.p,
        fixed_params.c,
        xi,
        tau,
        a,
        d_E
    ] for (xi, a, tau, d_E, delta_E, K_delta_E) in sampled_params]
end

# Define time span and initial conditions
tspan = (Float32(0.0), Float32(8.0))
y0 = Float32[
    4e8,  # T
    75.0, # I1
    0.0,  # I2
    0.0,  # V
    0.0,  # E
    0.0,  # El
    0.0,  # z[1]
    0.0,  # z[2]
    0.0   # z[3]
]

# Function to solve ODEs in serial
@everywhere function serial_LCTModel(tspan, y0, param_sets)
    sols = [solve_LCTModel(tspan, y0, params) for params in param_sets]
    return sols
end

@everywhere function pmap_LCTModel(tspan, y0, param_sets)
    sols = pmap(params -> solve_LCTModel(tspan, y0, params), param_sets)
    return sols
end

@everywhere function tmap_LCTModel(tspan, y0, param_sets)
    sols = Vector{Any}(undef, length(param_sets))
    Threads.@threads for i in 1:length(param_sets)
        sols[i] = solve_LCTModel(tspan, y0, param_sets[i])
    end
    return sols
end

@everywhere function test_methods_on_param_sets(param_sets, tspan, y0; profile=true)
    methods = Dict(
        :serial => serial_LCTModel,
        #:pmap   => pmap_LCTModel,
        :tmap   => tmap_LCTModel
    )

    results = Dict{Symbol, Any}()

    # Loop through each method
    for (method_name, method_func) in methods
        println("Testing method: $method_name")
        start_time = time()

        if profile
            println("Profiling method: $method_name")
            Profile.clear()  # Clear previous profiling data
            @profile sols = method_func(tspan, y0, param_sets)
        else
            sols = method_func(tspan, y0, param_sets)
        end

        end_time = time()
        total_time = end_time - start_time
        avg_time_per_sample = total_time / length(param_sets)
        avg_num_time_points = mean([length(sol.t) for sol in sols])

        # Store the results in the dictionary
        results[method_name] = (
            total_time=total_time,
            avg_time_per_sample=avg_time_per_sample,
            avg_num_time_points=avg_num_time_points
        )

        # Print profiling data summary to console (optional)
        if profile
            println("Profiling data for $method_name:")
            Profile.print()  # Print profiling data summary
        end
    end

    return results
end

# Generate parameter sets
seed = 12345
n_samples = 1000  # Number of samples
sampled_params = sample_parameters(seed, n_samples)
full_param_sets = create_full_parameter_sets(sampled_params)

# Run the benchmark
results = test_methods_on_param_sets(full_param_sets, tspan, y0; profile=false)
