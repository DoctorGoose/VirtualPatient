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

# T Cell Model with distinct linear chains
function LCTModel!(du, u, p, t, state_history::StateHistory)
    update!(state_history, t, u)

    beta, k, p_param, c, delta, xi, a, b, d_E, delta_E, K_delta_E, zeta, eta, K_I1, tau_memory, z1_len, z2_len  = p
    z1_len = Int(z1_len)  # Cast to integer
    z2_len = Int(z2_len) 

    # Unpack states
    T, I1, I2, V, CD8_E, CD8_M = u[1:6]
    z1_start, z2_start = 7, 7 + z1_len  
    z1 = u[z1_start:z1_start + z1_len - 1]
    z2 = u[z2_start:z2_start + z2_len - 1]

    # Retrieve delayed state
    CD8_E_tau = interpolate_delay(state_history, 5, t - tau_memory)  # 5th state (CD8_E)

    # Equations
    du[1] = -beta * T * V  # Target Cells
    du[2] = beta * T * V - k * I1  # Eclipse Cells
    du[3] = k * I1 - delta * I2 - delta_E * CD8_E * I2 / (K_delta_E + I2)  # Infected Cells
    du[4] = p_param * I2 - c * V  # Virus
    du[5] = a * z1[end] + b * z2[end] - d_E * CD8_E  # Effector T Cells
    du[6] = zeta * CD8_E_tau #- xi * I1 * CD8_M # Memory T Cells

    # Chain 1 dynamics
    du[z1_start] = xi * I1 - a * z1[1]
    du[z1_start + 1:z1_start + z1_len - 1] .= a .* (z1[1:end-1] .- z1[2:end])

    # Chain 2 dynamics
    du[z2_start] = I1 * CD8_M * eta - b * z2[1]
    du[z2_start + 1:z2_start + z2_len - 1] .= b .* (z2[1:end-1] .- z2[2:end])

    # Template for Chain 3+
    # z3_len = p[:z3]  # Length of Chain 3
    # z3_start = z2_start + z2_len
    # z3 = u[z3_start:z3_start + z3_len - 1]
    #
    # Chain 3 dynamics
    # du[z3_start] = <equation>
    # du[z3_start + 1:z3_start + z3_len - 1] .= a .* (z3[1:end-1] .- z3[2:end])
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
