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
    T, I1, I2, V, CD8E_early, CD8E_late, CD8E_tot, CD8M, inflammation = u[1:9]

    # Unpack parameters
    beta, k, p_param, c, delta, xi, K_E, a, d_E, delta_E, K_delta_E, zeta, eta, K_I1, tau_m, tau_e, alphaone, alphatwo = p

    # Retrieve delayed state
    CD8_E_tau = interpolate_delay(state_history, 7, t - tau_m)  # 7th state (CD8_total)
    E_I2_tau = interpolate_delay(state_history, 3, t - tau_e)  # 3rd state (I2)
    # Activation from memory -> effectors
    #activation = (xi*CD8_M * I1) / (I1 + K_I1)

    term1 = (xi*I2) / (K_E + CD8E_tot)
    term2 = eta * E_I2_tau * CD8E_tot
    # Equations
    du[1] = -beta * T * V  # Target Cells
    du[2] = beta * T * V - k * I1  # Eclipse Cells
    du[3] = k * I1 - delta * I2 - delta_E * CD8E_late * I2 / (K_delta_E + I2)  # Infected Cells
    du[4] = p_param * I2 - c * V  # Virus
    du[5] = term1 - d_E * CD8E_early  # Early Effector T Cells 
    du[6] = term2 - d_E * CD8E_late  # Late Effector T Cells
    du[7] = term1 + term2 - d_E * CD8E_tot  # total Effector T Cells 
    du[8] = zeta * CD8_E_tau ## Memory T Cells
    du[9] = alphaone * I1 + alphatwo * I2

end


function tmap_LCTModel(tspan, y0, param_sets)
    param_sets = isa(param_sets, AbstractVector) && isa(param_sets[1], AbstractVector) ? param_sets : [param_sets]
    sols = ThreadsX.map(params -> solve_LCTModel(tspan, y0, params), param_sets)
    return length(sols) == 1 ? sols[1] : sols
end

function solve_LCTModel(tspan, y0, params)
    state_history = StateHistory(y0)
    wrapper!(du, u, p, t) = LCTModel!(du, u, p, t, state_history)
    prob = ODEProblem(wrapper!, y0, tspan, params)

    isoutofdomain = (u, p, t) -> any(x -> x < -1e-3, u)
    sol = suppress_warnings(() -> solve(prob, TRBDF2(autodiff=false); reltol=1e-4, abstol=1e-5, 
                                         dtmax=1e-1, dtmin=1e-8, isoutofdomain=isoutofdomain))
    return (sol.t, hcat(sol.u...))
end
