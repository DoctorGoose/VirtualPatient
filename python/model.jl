
# Define a threshold function for states
function apply_state_threshold!(u, threshold)
    for i in eachindex(u)
        if abs(u[i]) < threshold
            u[i] = 0.0
        end
    end
end

# ODE model definition with threshold application on states before usage
function LCTModel!(du, u, p, t)
    # Apply state threshold to prevent small floating-point errors in states before use
    state_threshold = 1e-8  # You can adjust this based on your system's needs
    apply_state_threshold!(u, state_threshold)

    T, I1, I2, V, E, El = u[1], u[2], u[3], u[4], u[5], u[6]
    z = u[7:end]

    beta, k, delta, delta_E, K_delta_E, p_param, c, xi, tau, a, d_E = p

    # ODEs
    du[1] = -beta * T * V  # dT_dt
    du[2] = beta * T * V - k * I1  # dI1_dt
    du[3] = k * I1 - delta * I2 - (delta_E * E * I2) / (K_delta_E + I2)  # dI2_dt
    du[4] = p_param * I2 - c * V  # dV_dt
    du[5] = (xi / tau) * z[end] - d_E * E  # dE_dt
    du[6] = d_E * E # Lung T cells

    # Delayed variables
    stages = length(z)
    dz_dt = zeros(eltype(u), stages)
    dz_dt[1] = (1 / tau) * (a * I2 - z[1])
    for i in 2:stages
        dz_dt[i] = (1 / tau) * (z[i - 1] - z[i])
    end
    du[7:end] .= dz_dt
end

# Function to solve the model and return time points and solution values
function solve_LCTModel(tspan, y0, params)
    prob = ODEProblem(LCTModel!, y0, tspan, params)
    sol = solve(prob, AutoVern7(Rodas5()), reltol=1e-5, abstol=1e-6)
    return (sol.t, hcat(sol.u...))  # return (time, solution matrix)
end