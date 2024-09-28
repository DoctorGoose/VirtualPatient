using Distributed
using LinearAlgebra
using Random
using Statistics
using DifferentialEquations
using Logging
using SparseArrays
using Interpolations
using JSON
using Sundials

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
        beta = p[1]
        k = p[2]
        delta = p[3]
        delta_E = p[4]
        K_delta_E = p[5]
        p_param = p[6]
        c = p[7]
        xi = p[8]
        tau = p[9]
        a = p[10]
        d_E = p[11]

        # Precompute constants
        tau_inv = 1.0 / tau
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
    return nothing
end

function compute_sse(sol, patient_data)
    patient_data = Dict(patient_data)
    patient_id = patient_data["id"]

    println("Computing SSE for patient $(patient_id)")

    sse = 0.0
    sse_statewise = zeros(2)  # For 'V' and 'CD8TE'

    # Extract model time points and solutions
    t_model = sol.t
    y_model = transpose(hcat(sol.u...))

    # Patient data
    time_points = collect(patient_data["DAY"])
    data_V = collect(patient_data["V"])
    data_CD8TE = collect(patient_data["CD8TE"])

    # Handle missing data (NaNs)
    valid_indices_V = .!isnan.(data_V)
    valid_indices_CD8TE = .!isnan.(data_CD8TE)

    # Interpolate 'V'
    idx_V = 4  # Index for 'V' in y_model
    y_V_model = y_model[:, idx_V]
    itp_V = LinearInterpolation(t_model, y_V_model, extrapolation_bc=Line())
    interpolated_V = itp_V(time_points[valid_indices_V])

    # Compute SSE for 'V'
    log_data_V = log10.(clamp.(data_V[valid_indices_V], 1, Inf))
    log_model_V = log10.(clamp.(interpolated_V, 1, Inf))
    sse_V = sum((log_model_V .- log_data_V).^2)

    # Interpolate 'CD8TE'
    idx_CD8TE = 5  # Index for 'CD8TE' in y_model
    y_CD8TE_model = y_model[:, idx_CD8TE]
    itp_CD8TE = LinearInterpolation(t_model, y_CD8TE_model, extrapolation_bc=Line())
    interpolated_CD8TE = itp_CD8TE(time_points[valid_indices_CD8TE])

    # Compute SSE for 'CD8TE'
    log_data_CD8TE = log10.(clamp.(data_CD8TE[valid_indices_CD8TE], 1, Inf))
    log_model_CD8TE = log10.(clamp.(interpolated_CD8TE, 1, Inf))
    sse_CD8TE = sum((log_model_CD8TE .- log_data_CD8TE).^2)

    # Total SSE
    sse = sse_V + sse_CD8TE
    sse_statewise = [sse_V, sse_CD8TE]

    println("SSE for patient $(patient_id): V_sse = $(sse_V), CD8TE_sse = $(sse_CD8TE), total_sse = $(sse)")
    return sse, sse_statewise
end

function solve_LCTModel(tspan, y0, params_array)
    y0_float64 = convert(Vector{Float64}, y0)
    params_float64 = convert(Vector{Float64}, params_array)
    tspan_float64 = Float64.(tspan)

    solver = CVODE_BDF(method = :Newton, linear_solver = :GMRES)
    prob = ODEProblem(LCTModel!, y0_float64, tspan_float64, params_float64)
    sol = suppress_warnings(() -> solve(prob, solver; reltol=1e-5, abstol=1e-4))

    return sol
end

function tmap_LCTModel(tspan, y0, param_sets, patients_data)
    num_param_sets = size(param_sets, 1)
    num_patients = length(patients_data)
    total_results = num_param_sets * num_patients
    results = Vector{Any}(undef, total_results)

    println("Starting tmap_LCTModel with num_param_sets = $(num_param_sets), num_patients = $(num_patients)")

    Threads.@threads for idx in 1:num_param_sets
        params = param_sets[idx, :]
        println("Thread $(Threads.threadid()) - Solving model for parameter set idx = $(idx)")

        sol = solve_LCTModel(tspan, y0, params)

        if sol === nothing
            println("Thread $(Threads.threadid()) - Failed to solve model for idx = $(idx)")
            continue  # Skip to next parameter set
        else
            println("Thread $(Threads.threadid()) - Model solved for idx = $(idx)")
        end

        local_results = Vector{Any}(undef, num_patients)

        for (j, patient_data) in enumerate(patients_data)
            patient_id = patient_data["id"]
            sse, sse_statewise = compute_sse(sol, patient_data)
            println("Thread $(Threads.threadid()) - Computed SSE for patient $(patient_id): sse = $(sse), sse_statewise = $(sse_statewise)")

            result = Dict(
                "params" => params,
                "patient_id" => patient_id,
                "sse" => sse,
                "sse_statewise" => sse_statewise
            )
            local_results[j] = result
        end

        # Ensure thread-safe writing to the shared results array
        start_idx = (idx - 1) * num_patients + 1
        @inbounds for j in 1:num_patients
            results[start_idx + j - 1] = local_results[j]
        end
    end

    println("Finished tmap_LCTModel")
    return results
end