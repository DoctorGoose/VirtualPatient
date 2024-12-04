using DifferentialEquations
using Logging
using ThreadsX

function suppress_warnings(f)
    logger = ConsoleLogger(stderr, Logging.Error)  # Only show errors and above
    return with_logger(logger) do
        f()
    end
end

function LCTModel!(du, u, p, t)
    @inbounds begin
        # Unpack states
        T, I1, I2, V, E, El = u[1:6]
        z = u[7:end]

        # Unpack parameters
        beta, k, delta, delta_E, K_delta_E, p_param, c, xi, a, d_E = p

        if t < 0
            # No changes during the negative time
            du .= 0.0  # Set all derivatives to zero
        else
            # Normal ODEs for t >= 0
            # Precompute constant
            delta_E_term = delta_E * E * I2 / (K_delta_E + I2)

            # ODEs
            du[1] = -beta * T * V # Target Cells
            du[2] = beta * T * V - k * I1 # Eclipse Cells
            du[3] = k * I1 - delta * I2 - delta_E * E * I2 / (K_delta_E + I2) # Infected Cells
            du[4] = p_param * I2 - c * V # Virus
            du[5] = a * z[end] - d_E * E # T Cells
            du[6] = d_E * E # Exited T Cells (unused)

            # Delayed variables
            du[7] = xi * I1 - a * z[1] # First delay compartment
            du[8:end] .= a .* (z[1:end-1] .- z[2:end]) #nth compartment
        end
    end
    return nothing
end

function tmap_LCTModel(tspan, y0, param_sets)
    param_sets = isa(param_sets, AbstractVector) && isa(param_sets[1], AbstractVector) ? param_sets : [param_sets]
    sols = ThreadsX.map(params -> solve_LCTModel(tspan, y0, params), param_sets)
    return length(sols) == 1 ? sols[1] : sols
end

function solve_LCTModel(tspan, y0, params)
    prob = ODEProblem(LCTModel!, y0, tspan, params)
    isoutofdomain = (u, p, t) -> any(x -> x < -1e-3, u)
    sol = suppress_warnings(() -> solve(prob, TRBDF2(autodiff=true); reltol=1e-4, abstol=1e-5, 
                                         dtmax=1e-1, dtmin=1e-8, isoutofdomain=isoutofdomain))
    return (sol.t, hcat(sol.u...))
end

