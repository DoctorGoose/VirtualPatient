using DifferentialEquations

# ODE model definition
function LCTModel!(du, u, p, t)
    T, I1, I2, V, E = u[1], u[2], u[3], u[4], u[5]
    z = u[6:end]

    beta, k, delta, delta_E, K_delta_E, p_param, c, xi, tau, a, d_E = p

    # ODEs
    du[1] = -beta * T * V  # dT_dt
    du[2] = beta * T * V - k * I1  # dI1_dt
    du[3] = k * I1 - delta * I2 - (delta_E * E * I2) / (K_delta_E + I2)  # dI2_dt
    du[4] = p_param * I2 - c * V  # dV_dt
    du[5] = (xi / tau) * z[end] - d_E * E  # dE_dt

    # Delayed variables
    stages = length(z)
    dz_dt = zeros(eltype(u), stages)
    dz_dt[1] = (1 / tau) * (a * I2 - z[1])
    for i in 2:stages
        dz_dt[i] = (1 / tau) * (z[i - 1] - z[i])
    end
    du[6:end] .= dz_dt
end

# Function to solve the model and return time points and solution values
function solve_LCTModel(tspan, y0, params)
    prob = ODEProblem(LCTModel!, y0, tspan, params)
    sol = solve(prob, AutoVern7(Rodas5()), reltol=1e-5, abstol=1e-6)
    return (sol.t, hcat(sol.u...))  # No transpose needed, return as (time, solution matrix)
end
