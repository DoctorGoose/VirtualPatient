using DifferentialEquations

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

@everywhere function solve_LCTModel(tspan, y0, params, t_fill=[])
    prob = ODEProblem(LCTModel!, y0, tspan, params)
    sol = solve(prob, TRBDF2(autodiff=true), reltol=1e-6, abstol=1e-8, saveat=t_fill)
    return sol.t, sol.u
end

# Wrapper function for parallel execution using pmap
@everywhere function pmap_LCTModel(tspan, y0, params_list, t_fill)
    if size(params_list, 1) == 1
        solve_LCTModel(tspan, y0, params_list[1, :], t_fill)
    else
        return pmap(params -> solve_LCTModel(tspan, y0, params, t_fill), eachrow(params_list))
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

