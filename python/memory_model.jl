using DifferentialEquations
using ThreadsX
using Logging

function suppress_warnings(f)
    logger = ConsoleLogger(stderr, Logging.Error)  # Only show errors and above
    return with_logger(logger) do
        f()
    end
end

struct StateHistory
    history::Vector{Vector{Float64}}
end

function StateHistory(y0::Vector{Float64})
    StateHistory([vcat(0.0, y0)])
end

function update!(state_history::StateHistory, t::Float64, y::Vector{Float64})
    push!(state_history.history, vcat(t, y))
end

function interpolate_delay(state_history::StateHistory, state_idx::Int, t_delay::Float64)
    history = state_history.history
    if t_delay <= 0.0
        return history[1][state_idx + 1]  # Initial condition
    end

    idx = findlast(x -> x[1] <= t_delay, history)
    if idx === nothing || idx == length(history)
        return history[end][state_idx + 1]
    end

    t1, t2 = history[idx][1], history[idx + 1][1]
    y1, y2 = history[idx][state_idx + 1], history[idx + 1][state_idx + 1]

    return y1 + (t_delay - t1) * (y2 - y1) / (t2 - t1)
end

# Define the T Cell Model with delay
function LCTModel!(du, u, p, t, state_history::StateHistory)
    update!(state_history, t, u)

    # Unpack states
    T, I1, I2, V, CD8_E, CD8_M = u[1:6]
    z = u[7:end]

    # Unpack parameters
    beta, k, p_param, c, delta, xi, a, d_E, delta_E, K_delta_E, zeta, eta, K_I1, tau_memory = p

    # Retrieve delayed state
    CD8_E_tau = interpolate_delay(state_history, 4, t - tau_memory)  # 5th state (CD8_E)

    # Conditionally set beta*T*V term to zero if V < 1
    beta_T_V = V < 1 ? 0.0 : beta * T * V

    # Equations
    du[1] = -beta_T_V  # Target Cells
    du[2] = beta_T_V - k * I1  # Eclipse Cells
    du[3] = k * I1 - delta * I2 - delta_E * CD8_E * I2 / (K_delta_E + I2)  # Infected Cells
    du[4] = p_param * I2 - c * V  # Virus
    du[5] = a * z[end] + (eta * CD8_M * I1)/(I1 + K_I1) - d_E * CD8_E  # Effector T Cells (+  a * z[end])
    du[6] = zeta * CD8_E_tau #- eta * CD8_M * I1  # Memory T Cells

    # Delayed compartments
    du[7] = xi * I1 - a * z[1]
    du[8:end] .= a .* (z[1:end-1] .- z[2:end])
end


function solve_reinfection(tspan, y0, params)
    state_history = StateHistory(y0)
    wrapper!(du, u, p, t) = LCTModel!(du, u, p, t, state_history)
    prob = ODEProblem(wrapper!, y0, tspan, params)
    
    sol = solve(prob, TRBDF2(autodiff = false);
                reltol = 1e-4, abstol = 1e-5,
                dtmax = 1e-1, dtmin = 1e-8)
    return (sol.t, hcat(sol.u...))
end

const THRESHOLD = 1.5  # Threshold for zero-crossing detection

# Condition function for detecting near-zero crossing
function condition(u, t, integrator)
    # Check if any state is below the threshold
    (u[2] < THRESHOLD) || (u[3] < THRESHOLD) || (u[4] < THRESHOLD)
end

# Affect function to modify state and derivative
function affect!(integrator)
    for i in 2:4  # Indices of I1, I2, V
        if integrator.u[i] < THRESHOLD
            integrator.u[i] = 0.0  # Set state to zero
        end
    end
end

# Create the callback
cb = DiscreteCallback(condition, affect!)

# Wrapper to solve the problem with the callback
function solve_with_callback(tspan, y0, params)
    state_history = StateHistory(y0)
    wrapper!(du, u, p, t) = LCTModel!(du, u, p, t, state_history)
    prob = ODEProblem(wrapper!, y0, tspan, params)
    
    sol = solve(prob, TRBDF2(autodiff = false);
                callback = cb,
                reltol = 1e-4, abstol = 1e-5,
                dtmax = 1e-1, dtmin = 1e-8)
    return (sol.t, hcat(sol.u...))
end