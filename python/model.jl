# Julia ODE Model
using Distributed
using LinearAlgebra

blas_threads = 1
BLAS.set_num_threads(blas_threads)
println("Number of BLAS threads: ", BLAS.get_num_threads())
flush(stdout)
println("Number of threads: ", Threads.nthreads())
flush(stdout)

using LinearAlgebra
using Random
using Statistics
using DifferentialEquations
using Logging
using Profile
using SparseArrays
using Sundials

function suppress_warnings(f)
    logger = ConsoleLogger(stderr, Logging.Error)
    return with_logger(logger) do
        f()
    end
end

function LCTModel!(du, u, params, t)
    @inbounds begin
        # Unpack states
        T, I1, I2, V, E, El = u[1:6]
        z = u[7:end]

        # Unpack parameters
        beta, k, p, c, delta, xi, a, tau, d_E, delta_E, K_delta_E = params

        # Precompute constants
        tau_inv = 1.0 / tau
        xi_tau_inv = xi * tau_inv
        delta_E_term = delta_E * E * I2 / (K_delta_E + I2)

        # ODEs
        du[1] = -beta * T * V
        du[2] = beta * T * V - k * I1
        du[3] = k * I1 - delta * I2 - delta_E_term
        du[4] = p * I2 - c * V
        du[5] = xi_tau_inv * z[end] - d_E * E
        du[6] = d_E * E  # Lung T cells

        # Delayed variables
        du[7] = tau_inv * (a * I2 - z[1])  # dz_dt[1]
        du[8:end] .= tau_inv .* (z[1:end-1] .- z[2:end])  # dz_dt[2:end]
    end
    return nothing
end

function solve_LCTModel(tspan, y0, params)
    solver = CVODE_BDF(
        method = :Newton,
        linear_solver = :GMRES,
        #krylov_dim = 10,            
        max_nonlinear_iters = 10,   
        #max_convergence_failures = 3,

    )
    #solver=TRBDF2(autodiff=true)
    prob = ODEProblem(LCTModel!, y0, tspan, params)
    sol = suppress_warnings(() ->solve(prob, solver; reltol=1e-5, abstol=1e-4, dtmax=1e-1, dtmin=1e-8))
    #sol = solve(prob, solver; reltol=1e-4, abstol=1e-5, dtmax=1e-1, dtmin=1e-8)

    return (sol.t, hcat(sol.u...))  # return (time, solution matrix)
end

function serial_LCTModel(tspan, y0, param_sets)
    sols = [solve_LCTModel(tspan, y0, params) for params in param_sets]
    return sols
end

function tmap_LCTModel(tspan, y0, param_sets)
    sols = Vector{Any}(undef, length(param_sets))
    #print(param_sets)
    Threads.@threads for i in 1:length(param_sets)
        sols[i] = solve_LCTModel(tspan, y0, param_sets[i])
    end
    return sols
end

# Self-Benchmark
function sample_parameters(seed, n_samples)
    Random.seed!(seed)
    rng = Random.GLOBAL_RNG
    K_delta_E_range = (1e2, 1e6)

    param_samples = [(
        exp(rand(rng, Float64) * (log(K_delta_E_range[2]) - log(K_delta_E_range[1])) + log(K_delta_E_range[1]))
    ) for _ in 1:n_samples]
    return param_samples
end

function create_full_parameter_sets(sampled_params)
    fixed_params = (
        beta=1.0888e-4,
        k=4.0,
        p=0.02978,
        c=13.934,
        delta=0.96,
        xi=0.1198,
        a=0.3615,
        tau=1.38,
        d_E=1.25,
        delta_E=8.939,
    )
    return [Float64[
        fixed_params.beta,
        fixed_params.k,
        fixed_params.p,
        fixed_params.c,
        fixed_params.delta,
        fixed_params.xi,
        fixed_params.a,
        fixed_params.tau,
        fixed_params.d_E,
        fixed_params.delta_E,
        K_delta_E
    ] for K_delta_E in sampled_params]
end

function test_methods_on_param_sets(param_sets, tspan, y0; profile=true)
    methods = Dict(
        #:serial => serial_LCTModel,
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

        results[method_name] = (
            total_time=total_time,
        )

        if profile
            println("Profiling data for $method_name:")
            Profile.print()
        end
    end

    return results
end

function run_self_test()
    tspan = (Float64(0.0), Float64(8.0))
    y0 = Float64[
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

    seed = 12345
    n_samples = 100  
    sampled_params = sample_parameters(seed, n_samples)
    full_param_sets = create_full_parameter_sets(sampled_params)
    results = test_methods_on_param_sets(full_param_sets, tspan, y0; profile=false)

    println("Test completed with results: ", results)
end

#if "--run-test" in ARGS
#    println("Running self-test...")
#    run_self_test()
#end