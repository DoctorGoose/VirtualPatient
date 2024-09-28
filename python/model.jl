# ODE model definition with threshold application on states before usage
function LCTModel!(du, u, p, t)
    @inbounds begin
        # Unpack states
        T = u[1]; I1 = u[2]; I2 = u[3]; V = u[4]; E = u[5]; El = u[6]
        z = u[7:end]

        # Unpack parameters
        beta = p[1]; k = p[2]; delta = p[3]; delta_E = p[4]; K_delta_E = p[5]
        p_param = p[6]; c = p[7]; xi = p[8]; tau = p[9]; a = p[10]; d_E = p[11]

        # Precompute constants
        tau_inv = 1 / tau
        xi_tau_inv = xi * tau_inv
        delta_E_term = delta_E * E * I2 / (K_delta_E + I2)

        # ODEs
        du[1] = -beta * T * V
        du[2] = beta * T * V - k * I1
        du[3] = k * I1 - delta * I2 - delta_E_term
        du[4] = p_param * I2 - c * V
        du[5] = xi_tau_inv * z[end] - d_E * E
        du[6] = d_E * E  # Lung T cells

        # Delayed variables
        du[7] = tau_inv * (a * I2 - z[1])  # dz_dt[1]
        du[8:end] .= tau_inv .* (z[1:end-1] .- z[2:end])  # dz_dt[2:end]
    end
end

# Function to solve the model and return time points and solution values
function solve_LCTModel(tspan, y0, params)
    prob = ODEProblem(LCTModel!, y0, tspan, params)
    sol = solve(prob, TRBDF2(autodiff=true), reltol=1e-5, abstol=1e-6)
    return (sol.t, hcat(sol.u...))  # return (time, solution matrix)
end