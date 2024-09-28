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

@everywhere function LCTModel!(du, u, p, t)
    @inbounds begin
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
    # Ensure y0 and params are Vector{Float32}
    y0_float32 = Vector{Float32}(y0)
    params_float32 = Vector{Float32}(params)
    tspan_float32 = (Float32(tspan[1]), Float32(tspan[2]))

    prob = ODEProblem(LCTModel!, y0_float32, tspan_float32, params_float32)
    sol = solve(prob, TRBDF2(autodiff=true); reltol=1e-5, abstol=1e-4)

    return (sol.t, hcat(sol.u...))  # return (time, solution matrix)
end

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