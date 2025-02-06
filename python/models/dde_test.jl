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

function lorenz_with_delay!(du, u, p, t, state_history::StateHistory)
    update!(state_history, t, u)

    # Unpack parameters
    sigma, rho, beta, tau1, tau2 = p

    # Retrieve delayed states
    X_tau1 = interpolate_delay(state_history, 1, t - tau1)
    X_tau2 = interpolate_delay(state_history, 1, t - tau2)

    # Equations
    dX = sigma * (u[2] - X_tau1)
    dY = X_tau1 * (rho - u[3]) - u[2]
    dZ = X_tau2 * u[2] - beta * u[3]

    du[1] = dX
    du[2] = dY
    du[3] = dZ
end

using DifferentialEquations

# Parameters
p = [10.0, 28.0, 8 / 3, 2.0, 4.0]  # Convert parameters to a vector
y0 = [1.0, 1.0, 1.0]  # Initial state
tspan = (0.0, 10.0)
state_history = StateHistory(y0)  # Initialize state history

# Wrapper function for ODEProblem
function dde_wrapper!(du, u, p, t)
    lorenz_with_delay!(du, u, p, t, state_history)
end

# Define and solve the problem
prob = ODEProblem(dde_wrapper!, y0, tspan, p)
sol = solve(prob, Tsit5(), saveat=0.01)
