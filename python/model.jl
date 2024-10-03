using Distributed
using LinearAlgebra

blas_threads = 1
BLAS.set_num_threads(blas_threads)

using Random
using Statistics
using DifferentialEquations
using Logging
using Profile
using SparseArrays
#using Sundials

function suppress_warnings(f)
    logger = ConsoleLogger(stderr, Logging.Error)  # Only show errors and above
    return with_logger(logger) do
        f()
    end
end

function LCTModel!(du, u, p, t)
    @inbounds begin
        # Unpack states
        T, I1, I2, V, E, El = u[1:6]
        z = u[7:end]

        # Unpack parameters
        beta, k, delta, delta_E, K_delta_E, p_param, c, xi, tau, a, d_E = p

        # Precompute constants
        tau_inv = 1.0 / tau
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

function jac_sparsity!(S, u, p, t)
    @inbounds begin
        # Non-zero sparsity entries based on the system of ODEs
        
        # T (du[1]) depends on T (u[1]) and V (u[4])
        S[1, 1] = true  # T affects dT/dt
        S[1, 4] = true  # V affects dT/dt

        # I1 (du[2]) depends on T (u[1]), I1 (u[2]), and V (u[4])
        S[2, 1] = true  # T affects dI1/dt
        S[2, 2] = true  # I1 affects dI1/dt
        S[2, 4] = true  # V affects dI1/dt

        # I2 (du[3]) depends on I1 (u[2]), I2 (u[3]), and E (u[5])
        S[3, 2] = true  # I1 affects dI2/dt
        S[3, 3] = true  # I2 affects dI2/dt
        S[3, 5] = true  # E affects dI2/dt

        # V (du[4]) depends on I2 (u[3]) and V (u[4])
        S[4, 3] = true  # I2 affects dV/dt
        S[4, 4] = true  # V affects dV/dt

        # E (du[5]) depends on z[end] (u[9]) and E (u[5])
        S[5, 9] = true  # z[end] affects dE/dt
        S[5, 5] = true  # E affects dE/dt

        # El (du[6]) depends on E (u[5])
        S[6, 5] = true  # E affects dEl/dt

        # z[1] (du[7]) depends on I2 (u[3]) and z[1] (u[7])
        S[7, 3] = true  # I2 affects dz[1]/dt
        S[7, 7] = true  # z[1] affects dz[1]/dt

        # z[2] (du[8]) depends on z[1] (u[7]) and z[2] (u[8])
        S[8, 7] = true  # z[1] affects dz[2]/dt
        S[8, 8] = true  # z[2] affects dz[2]/dt

        # z[end] (du[9]) depends on z[end-1] (u[8]) and z[end] (u[9])
        S[9, 8] = true  # z[end-1] affects dz[end]/dt
        S[9, 9] = true  # z[end] affects dz[end]/dt
    end
end

function tmap_LCTModel(tspan, y0, param_sets)
    # If a single parameter set is passed, wrap it in an array for consistent handling
    if !isa(param_sets, AbstractVector) || (isa(param_sets, AbstractVector) && !isa(param_sets[1], AbstractVector))
        param_sets = [param_sets]
    end

    sols = Vector{Any}(undef, length(param_sets))
    Threads.@threads for i in 1:length(param_sets)
        sols[i] = solve_LCTModel(tspan, y0, param_sets[i])
    end

    # Return results: single solution if only one parameter set, otherwise return all solutions
    return length(sols) == 1 ? sols[1] : sols
end

function solve_LCTModel(tspan, y0, params)
    prob = ODEProblem(LCTModel!, y0, tspan, params)
    sol = suppress_warnings(() -> solve(prob, TRBDF2(autodiff=true); reltol=1e-3, abstol=1e-2, dtmax=1e-1, dtmin=1e-6))

    return (sol.t, hcat(sol.u...))
end

function solve_LCTModel_jac(tspan, y0, params)
    S = spzeros(9, 9)
    jac_sparsity!(S, y0, params, tspan[1])
    f = ODEFunction(LCTModel!, jac_prototype=S)
    prob = ODEProblem(f, y0, tspan, params)
    sol = suppress_warnings(() -> solve(prob, TRBDF2(autodiff=true); reltol=1e-3, abstol=1e-2))

    return (sol.t, hcat(sol.u...))
end

#function cvode_LCTModel(tspan, y0, params_array)

    #solver = CVODE_BDF(
    #    method = :Newton,
    #    linear_solver = :GMRES,
    #    krylov_dim = 10,            
    #    max_nonlinear_iters = 10,   
    #    max_convergence_failures = 3,
    #)
    #prob = ODEProblem(LCTModel!, y0, tspan, params_array)
    #sol = suppress_warnings(() ->solve(prob, solver; reltol=1e-3, abstol=1e-2, dtmax=1e-1, dtmin=1e-8))

    #return (sol.t, hcat(sol.u...))
#end

