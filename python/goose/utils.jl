using DifferentialEquations
using ThreadsX
using Logging

# Run function `f` with a logger that only shows errors.
function suppress_warnings(f::Function)
    logger = ConsoleLogger(stderr, Logging.Error)
    return with_logger(logger) do
        f()
    end
end

# Store times separately from state vectors for fast interpolation.
struct StateHistory
    times::Vector{Float64}
    states::Vector{Vector{Float64}}
end

StateHistory(y0::Vector{Float64}) = StateHistory([0.0], [copy(y0)])

# Append the current time and state vector to the history.
function update!(state_history::StateHistory, t::Float64, y::Vector{Float64})
    push!(state_history.times, t)
    push!(state_history.states, copy(y))
end

# Interpolate the delayed state value for a given state index at time `t_delay`.
# If `t_delay` is before time zero or beyond the history, returns the appropriate boundary value.
function interpolate_delay(state_history::StateHistory, state_idx::Int, t_delay::Float64)
    times = state_history.times
    states = state_history.states
    if t_delay <= 0.0
        @inbounds return states[1][state_idx]
    end
    # Use binary search for the last time <= t_delay.
    idx = searchsortedlast(times, t_delay)
    if idx == length(times)
        @inbounds return states[end][state_idx]
    end
    @inbounds begin
        t1 = times[idx]
        t2 = times[idx+1]
        y1 = states[idx][state_idx]
        y2 = states[idx+1][state_idx]
        return y1 + (t_delay - t1) * (y2 - y1) / (t2 - t1)
    end
end

# ODE solver that accepts a model function `model_func!` with signature
# `(du, u, p, t, state_history)` and returns a tuple `(t, y)`, where `y` is a matrix of solution values.
function solve_ode_model(model_func!, tspan, y0, params)
    state_history = StateHistory(y0)
    wrapper!(du, u, p, t) = model_func!(du, u, p, t, state_history)
    prob = ODEProblem(wrapper!, y0, (tspan[1], tspan[end]), params)
    
    isoutofdomain = (u, p, t) -> any(x -> x < -1e-3, u)
    
    # Solve the ODE, saving at the time points given in tspan.
    sol = suppress_warnings(() -> solve(prob, TRBDF2(autodiff=false);
                                          reltol=1e-4, abstol=1e-5,
                                          dtmax=1e-1, dtmin=1e-8,
                                          saveat=tspan,
                                          isoutofdomain=isoutofdomain))
    
    @info "Solution computed: length(sol.t) = $(length(sol.t))"
    return (sol.t, hcat(sol.u...))
end

# Applies `solve_ode_model` in parallel over a collection of parameter sets using ThreadsX.
function tmap_model(model_func!, tspan, y0, param_sets)
    param_sets = (isa(param_sets, AbstractVector) && isa(param_sets[1], AbstractVector)) ? param_sets : [param_sets]
    return ThreadsX.map(params -> solve_ode_model(model_func!, tspan, y0, params), param_sets)
end