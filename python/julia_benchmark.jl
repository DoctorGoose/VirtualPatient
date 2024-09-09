
using DifferentialEquations
using Distributed
using Random
using BenchmarkTools

@everywhere using DifferentialEquations

# ODE model 
@everywhere function LCTModel!(du, u, p, t)
    # Apply state threshold to prevent small floating-point errors in states
    state_threshold = 1e-8
    @inbounds for i in eachindex(u)
        u[i] = abs(u[i]) < state_threshold ? 0.0 : u[i]
    end

    # Extract parameters
    T, I1, I2, V, E, El = u[1], u[2], u[3], u[4], u[5], u[6]
    z = @view u[7:end]  # Use view to avoid copying the array

    beta, k, delta, delta_E, K_delta_E, p_param, c, xi, tau, a, d_E = p

    # ODEs 
    du[1] = -beta * T * V               # dT_dt
    du[2] = beta * T * V - k * I1        # dI1_dt
    du[3] = k * I1 - delta * I2 - (delta_E * E * I2) / (K_delta_E + I2)  # dI2_dt
    du[4] = p_param * I2 - c * V         # dV_dt
    du[5] = (xi / tau) * z[end] - d_E * E # dE_dt
    du[6] = d_E * E                      # Lung T cells

    # Delayed variables 
    stages = length(z)
    @inbounds begin
        du[7] = (1 / tau) * (a * I2 - z[1])  # First delayed stage
        for i in 2:stages
            du[i + 6] = (1 / tau) * (z[i - 1] - z[i])  # Updating du[7:end]
        end
    end
end

# Function to solve the model and return time points and solution values
@everywhere function solve_LCTModel(tspan, y0, params, t_fill)
    println("Solving for tspan: ", tspan)
    println("Additional time points (t_fill): ", t_fill)

    # Define the problem
    prob = ODEProblem(LCTModel!, y0, tspan, params)

    # Solve the problem with the given settings
    sol = solve(prob, TRBDF2(autodiff=true), reltol=1e-5, abstol=1e-6, saveat=t_fill)

    # Print the evaluated time points from the solver
    println("Evaluated time points (sol.t): ", sol.t)

    # Return the time points and the solution matrix
    return (sol.t, hcat(sol.u...))  # return (time, solution matrix)
end

# Wrapper function for parallel execution using pmap
@everywhere function pmap_LCTModel(tspan, y0, params_list, t_fill=[])
    println("Running pmap_LCTModel with t_fill: ", t_fill)

    if size(params_list, 1) == 1
        # Single parameter set, solve serially
        t_values, y_values = solve_LCTModel(tspan, y0, params_list[1, :], t_fill)
        return (t_values, y_values)
    else
        # Multiple parameter sets, solve in parallel
        solutions = pmap(params -> solve_LCTModel(tspan, y0, params, t_fill), eachrow(params_list))

        # Extract time values and solution matrices from the results
        t_values_list = [sol[1] for sol in solutions]
        y_values_list = [sol[2] for sol in solutions]

        # Print the time matrix for debugging
        println("t_values_list: ", t_values_list)

        return (t_values_list, y_values_list)
    end
end

# Define a threshold function for states
@everywhere function apply_state_threshold!(u, threshold)
    for i in eachindex(u)
        if abs(u[i]) < threshold
            u[i] = 0.0
        end
    end
end


### Benchmmarking solvers ###
# Function to sample parameters uniformly in log space
@everywhere function sample_parameters(seed, n_samples)
    Random.seed!(seed)
    rng = Random.GLOBAL_RNG
    xi_range = (1e-2, 1.0)
    a_range = (1e-1, 1.0)
    tau_range = (1e-1, 5.0)
    d_E_range = (0.5, 10.0)
    delta_E_range = (1e-1, 1e2)
    K_delta_E_range = (1e2, 1e6)

    param_samples = [(
        exp(rand(rng) * (log(xi_range[2]) - log(xi_range[1])) + log(xi_range[1])),
        exp(rand(rng) * (log(a_range[2]) - log(a_range[1])) + log(a_range[1])),
        exp(rand(rng) * (log(tau_range[2]) - log(tau_range[1])) + log(tau_range[1])),
        exp(rand(rng) * (log(d_E_range[2]) - log(d_E_range[1])) + log(d_E_range[1])),
        exp(rand(rng) * (log(delta_E_range[2]) - log(delta_E_range[1])) + log(delta_E_range[1])),
        exp(rand(rng) * (log(K_delta_E_range[2]) - log(K_delta_E_range[1])) + log(K_delta_E_range[1]))
    ) for _ in 1:n_samples]
    return param_samples
end

fixed_params = (T0=4e8, I10=75, beta=1.0888e-4, k=4, p=0.02978, c=13.934, delta=0.96, E0=3.47e5)

# Combine fixed parameters with sampled parameters and flatten into a vector
@everywhere function create_full_parameter_sets(sampled_params)
    return [vcat(fixed_params.T0, fixed_params.I10, fixed_params.beta, fixed_params.k, fixed_params.p,
                 fixed_params.c, fixed_params.delta, xi, tau, a, d_E, delta_E, K_delta_E) 
            for (xi, a, tau, d_E, delta_E, K_delta_E) in sampled_params]
end

# List of solvers to test
solver_methods = [
    #Tsit5(),             # Non-stiff, explicit Runge-Kutta
    #Vern7(),             # Non-stiff, 7th order
    #Rodas5(),            # Stiff, Rosenbrock
    #AutoVern7(TRBDF2()), # Automatic switching
    TRBDF2()             # Stiff, Implicit BDF
]

# Define time span and initial conditions
tspan = (0.0, 8.0)
y0 = [4e8, 75.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Initial conditions

using Base: sleep

# Helper function to enforce timeout on long-running operations
function run_with_timeout(f, timeout_secs)
    result = nothing
    timed_out = false
    
    t = Timer(timeout_secs) do
        timed_out = true
        throw(InterruptException("Execution timed out"))
    end
    
    try
        result = f()  # Run the provided function
    catch e
        if isa(e, InterruptException)
            println("Function timed out!")
            result = NaN  # Return NaN if the function times out
        else
            rethrow(e)  # Rethrow any other exception
        end
    finally
        close(t)  # Close the timer
    end
    
    return result
end
using Statistics  # Import for mean and std functions

function test_solvers_on_param_sets(param_sets, solver_methods, tspan, y0)
    results = Dict{Symbol, Any}()
    total_samples = length(param_sets)
    
    for solver in solver_methods
        solver_name = Symbol(solver)  # Convert solver to Symbol for dictionary key
        println("Testing solver: ", solver_name)

        times = Float64[]
        
        for (i, params) in enumerate(param_sets)
            if i % 25 == 0
                println("Processing sample $i of $total_samples")
            end
            
            # Wrap the solver and param in a function for use with the timeout
            solver_func = let solver=solver, params=params  # Capture solver and params
                () -> begin
                    time_taken = @elapsed pmap_LCTModel(tspan, y0, [params])
                    return time_taken
                end
            end

            # Run the solver function with a timeout of 30 seconds
            time_taken = run_with_timeout(solver_func, 30)  # Timeout of 30 seconds
            
            if isnan(time_taken)
                println("Sample $i timed out for solver $solver_name")
            end

            push!(times, time_taken)
        end

        # Calculate and print the statistics for the solver's times
        min_time = minimum(times)
        max_time = maximum(times)
        mean_time = mean(times)
        std_time = std(times)
        
        println("Solver: $solver_name")
        println("Min time: $min_time seconds")
        println("Max time: $max_time seconds")
        println("Mean time: $mean_time seconds")
        println("Std time: $std_time seconds")
        
        results[solver_name] = times  # Use solver_name (Symbol) as the dictionary key
    end
    return results
end

# Generate 100 parameter sets with fixed seed
seed = 12345
n_samples = 10
sampled_params = sample_parameters(seed, n_samples)
full_param_sets = create_full_parameter_sets(sampled_params)

# Benchmark different solvers using the pmap_LCTModel function
solver_results = test_solvers_on_param_sets(full_param_sets, solver_methods, tspan, y0)