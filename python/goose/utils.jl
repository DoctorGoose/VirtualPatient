# julia utilities
using DifferentialEquations
using ThreadsX
using Logging

# Runs `f` with a logger that only shows errors (and above).
function suppress_warnings(f::Function)
    logger = ConsoleLogger(stderr, Logging.Error)
    return with_logger(logger) do
        f()
    end
end

# Keeps a history of states (each state vector prepended with its time) for delay interpolation.
struct StateHistory
    history::Vector{Vector{Float64}}
end

StateHistory(y0::Vector{Float64}) = StateHistory([vcat(0.0, y0)])


# Appends the current time and state vector to the history.
function update!(state_history::StateHistory, t::Float64, y::Vector{Float64})
    push!(state_history.history, vcat(t, y))
end


# Interpolates the delayed state value for a given state index at time `t_delay`.
# If `t_delay` is before time zero or beyond the history, returns the appropriate boundary value.
function interpolate_delay(state_history::StateHistory, state_idx::Int, t_delay::Float64)
    history = state_history.history
    if t_delay <= 0.0
        return history[1][state_idx + 1]  # initial condition
    end
    idx = findlast(x -> x[1] <= t_delay, history)
    if idx === nothing || idx == length(history)
        return history[end][state_idx + 1]
    end
    t1, t2 = history[idx][1], history[idx + 1][1]
    y1, y2 = history[idx][state_idx + 1], history[idx + 1][state_idx + 1]
    return y1 + (t_delay - t1) * (y2 - y1) / (t2 - t1)
end

# ODE solver that accepts a model function `model_func!` with signature
#`(du, u, p, t, state_history)` and returns a tuple `(t, y)`, where `y` is a matrix of solution values.
function solve_ode_model(model_func!, tspan, y0, params)
    state_history = StateHistory(y0)
    wrapper!(du, u, p, t) = model_func!(du, u, p, t, state_history)
    prob = ODEProblem(wrapper!, y0, tspan, params)
    isoutofdomain = (u, p, t) -> any(x -> x < -1e-3, u)
    sol = suppress_warnings(() -> solve(prob, TRBDF2(autodiff=false);
                                          reltol=1e-4, abstol=1e-5,
                                          dtmax=1e-1, dtmin=1e-8,
                                          isoutofdomain=isoutofdomain))
    return (sol.t, hcat(sol.u...))
end

# Applies `solve_ode_model` in parallel over a collection of parameter sets using ThreadsX.
# If only one parameter set is provided, returns its solution (still sometimes faster than running without ThreadsX)
function tmap_model(model_func!, tspan, y0, param_sets)
    param_sets = (isa(param_sets, AbstractVector) && isa(param_sets[1], AbstractVector)) ? param_sets : [param_sets]
    sols = ThreadsX.map(params -> solve_ode_model(model_func!, tspan, y0, params), param_sets)
    return length(sols) == 1 ? sols[1] : sols
end
