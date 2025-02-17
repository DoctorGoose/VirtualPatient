# Primary Infection Model
include("../goose/utils.jl")

function LCTModel!(du, u, p, t, state_history)
    update!(state_history, t, u)
    # Unpack states
    T, I1, I2, V, CD8_E, CD8_M = u[1:6]
    z = u[7:end]

    # Unpack parameters (14 parameters expected)
    beta, k, p_param, c, delta, xi, a, d_E, delta_E, K_delta_E, zeta, eta, K_I1, tau_memory = p

    # Retrieve delayed state (CD8_E at time t - tau_memory)
    CD8_E_tau = interpolate_delay(state_history, 5, t - tau_memory)

    # Model equations
    du[1] = -beta * T * V
    du[2] = beta * T * V - k * I1
    du[3] = k * I1 - delta * I2 - delta_E * CD8_E * I2 / (K_delta_E + I2)
    du[4] = p_param * I2 - c * V
    du[5] = a * z[end] - d_E * CD8_E
    du[6] = zeta * CD8_E_tau
    # Delayed compartments
    du[7] = xi * I1 - a * z[1]
    du[8:end] .= a .* (z[1:end-1] .- z[2:end])
end

# Wrapper for parallel solving 
function tmap_LCTModel(tspan, y0, param_sets)
    return tmap_model(LCTModel!, tspan, y0, param_sets)
end
