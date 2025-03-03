include("../goose/utils.jl")

# Reinfection Model
function ReinfectionModel!(du, u, p, t, state_history)
    update!(state_history, t, u)

    @inbounds begin
        T, I1, I2, V, CD8_E, CD8_M, CD8_MP, CD8_EP = u[1:8]
    end
    z = @inbounds u[9:end]

    # Unpack parameters (15 parameters expected)
    beta, k, p_param, c, delta, xi, a, d_E, delta_E, K_delta_E, zeta, eta, K_I1, tau_memory, damp = p

    # Retrieve delayed state (CD8_EP is u[5]) at time t - tau_memory.
    CD8_EP_tau = interpolate_delay(state_history, 8, t - tau_memory)

    # Damp (prevent new cell infections) if virus rebounds  
    if t > damp && V < 10
        du[1] = 0         # Target cells held constant
        du[2] = -k * I1   # allow Eclipse to form Infected
    else
        du[1] = -beta * T * V
        du[2] = beta * T * V - k * I1
    end

    # Core dynamics
    @inbounds begin
        du[3] = k * I1 - delta * I2 - delta_E * CD8_E * I2 / (K_delta_E + I2)
        du[4] = p_param * I2 - c * V
        du[5] = a * z[end] + (eta * CD8_MP * I1) / (K_I1 + I1) - d_E * CD8_E # Total effectors (killing)
        du[6] = zeta * CD8_EP_tau # Forming new memory only from primary response effectors
        du[7] = - (eta * CD8_MP * I1) / (K_I1 + I1) # Pre-existing memory pool(0 in naive)
        du[8] = a * z[end] - d_E * CD8_EP # Primary response effectors (memory-eligible)
        # Delayed compartments
        du[9] = xi * I1 - a * z[1]
        du[10:end] .= a .* (z[1:end-1] .- z[2:end])
    end
end

# Wrapper for parallel solving
function tmap_ReinfectionModel(tspan, y0, param_sets)
    return tmap_model(ReinfectionModel!, tspan, y0, param_sets)
end
