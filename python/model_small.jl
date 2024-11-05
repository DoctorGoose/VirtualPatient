using DifferentialEquations
using Logging
using ThreadsX

function suppress_warnings(f)
    logger = ConsoleLogger(stderr, Logging.Error)  # Only show errors and above
    return with_logger(logger) do
        f()
    end
end

function SmallModel!(du, u, p, t)
    @inbounds begin
        # Unpack states
        T, I1, I2, V = u[1:4]

        # Unpack parameters
        beta, k, delta, delta_E, K_delta_E, p_param, c, xi, a, d_E = p

        # ODEs
        du[1] = -beta * T * V
        du[2] = beta * T * V - k * I1
        du[3] = k * I1 - delta * I2 #- delta_E_term
        du[4] = p_param * I2 - c * V

        end
    end
    return nothing
end

function tmap_SmallModel(tspan, y0, param_sets)
    param_sets = isa(param_sets, AbstractVector) && isa(param_sets[1], AbstractVector) ? param_sets : [param_sets]
    sols = ThreadsX.map(params -> solve_SmallModel(tspan, y0, params), param_sets)
    return length(sols) == 1 ? sols[1] : sols
end

function solve_SmallModel(tspan, y0, params)
    prob = ODEProblem(SmallModel!, y0, tspan, params)
    isoutofdomain = (u, p, t) -> any(x -> x < -1e-3, u)
    sol = suppress_warnings(() -> solve(prob, TRBDF2(autodiff=true); reltol=1e-6, abstol=1e-5, 
                                         dtmax=1e-1, dtmin=1e-12, isoutofdomain=isoutofdomain))
    return (sol.t, hcat(sol.u...))
end

