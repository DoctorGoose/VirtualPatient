import re, time, os, shutil, tempfile
from collections import deque
import numpy as np
import seaborn as sns
import pandas as pd
import sqlite3
from scipy.optimize import OptimizeResult, OptimizeWarning, show_options
from scipy.optimize import differential_evolution, shgo, dual_annealing, basinhopping, direct, brute, minimize
import warnings
import copy

# Julia and model setup
# TODO pass model location (+handle?) as args
os.environ["JULIA_NUM_THREADS"] = "4"
from julia.api import Julia
julia = Julia(sysimage="../sysimage_env/sysimage.so")
from julia import Main
Main.include("../models/Primary_model.jl")
Main.include("../models/Reinfection_model_partial.jl") 

### Generally useful (infrastructure) functions ###
def sci_format(x, pos):
    coeff, exp = f"{x:.0e}".split("e")
    exp = int(exp)  # Convert exponent to an integer
    return f"{coeff}e{exp:+d}"  # Force a single-digit exponent when applicable

def format_fit_params(fit_parameters):
    if isinstance(fit_parameters, list) and len(fit_parameters) > 1:
        fit_params_str = " ".join(fit_parameters)
    else:
        fit_params_str = fit_parameters[0] if isinstance(fit_parameters, list) else fit_parameters

    # Dictionary mapping Greek letter names to glyphs
    greek_letters = {
        "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ",
        "epsilon": "ε", "zeta": "ζ", "eta": "η", "theta": "θ",
        "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ",
        "nu": "ν", "xi": "ξ", "omicron": "ο", "pi": "π",
        "rho": "ρ", "sigma": "σ", "tau": "τ", "upsilon": "υ",
        "phi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω"
    }

    # For each Greek letter name, replace it if it’s not preceded/followed by [A-Za-z].
    # This allows underscores or string boundaries on either side.
    for name, glyph in greek_letters.items():
        fit_params_str = re.sub(
            rf"(?<![A-Za-z]){name}(?![A-Za-z])",
            glyph,
            fit_params_str
        )

    # Remove underscores after replacing
    fit_params_str = fit_params_str.replace("_", "")

    return fit_params_str


def read_excel(filename):
    """Reads an Excel file that may be locked and returns a DataFrame."""
    temp_fd, temp_path = tempfile.mkstemp(suffix=".xlsx")
    os.close(temp_fd)  # Close the handle immediately

    try: # Attempt to copy the file even if it's locked
        shutil.copy2(filename, temp_path)  
        df = pd.read_excel(temp_path)  # Read the temporary copy 
        return df
    
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    
    finally: # Remove the temporary file 
        if os.path.exists(temp_path):
            os.remove(temp_path)

### Optimization ###

class NoOpMinimizer:
    def __call__(self, x):
        return x
    
class TimeManager:
    def __init__(self):
        self.start_time = time.time()

    def check_timeout(self, timeout: int) -> bool:
        return (time.time() - self.start_time) > timeout

    def reset_start_time(self):
        self.start_time = time.time()

    def get_elapsed_time(self) -> float:
        return time.time() - self.start_time

## ASA Optimizer ##

def vfsa_accprob(curr_cost, new_cost, temp_acc):
    exponent = np.clip((new_cost - curr_cost) / temp_acc, -500, 500)
    return 1 / (1 + np.exp(exponent))

def vfsa_gen_step(dim, log_lb, log_ub, temp_gen, rng=None):
    if rng is None: 
        rng = np.random.default_rng()

    uni = rng.random(dim)
    base = 1 + 1 / (temp_gen + 1e-10)  # Small value added to avoid division by zero
    exponent = 2 * uni - 1
    rnd = np.sign(uni - 0.5) * temp_gen * (base**np.abs(exponent) - 1)
    return (log_ub - log_lb) * rnd

def vfsa_gen_params(curr_params, dim, log_lb, log_ub, temp_gen, rng=None):
    if rng is None: 
        rng = np.random.default_rng()

    log_params = np.log10(curr_params)
    flag1 = True

    while flag1:
        # Generate a log step
        log_step = vfsa_gen_step(dim, log_lb, log_ub, temp_gen, rng)
        new_log_params = log_params + log_step

        # Check if all new parameters are within bounds
        if np.all(new_log_params >= log_lb) and np.all(new_log_params <= log_ub):
            # If within bounds, convert back to linear scale and return
            par = 10 ** new_log_params
            flag1 = False
        else:
            # If any parameter is out of bounds, handle each one individually
            for i in range(dim):
                if new_log_params[i] < log_lb[i] or new_log_params[i] > log_ub[i]:
                    flag2 = True
                    while flag2:
                        # Generate a new step for the out-of-bounds parameter
                        log_step = vfsa_gen_step(dim, log_lb, log_ub, temp_gen, rng)
                        new_log_params[i] = log_params[i] + log_step[i]

                        # Check if it's now within bounds
                        if log_lb[i] <= new_log_params[i] <= log_ub[i]:
                            flag2 = False

            # Once all parameters are adjusted, convert to linear scale and exit loop
            par = 10 ** new_log_params
            flag1 = False

    return par
        
def vfsa_generinitpoint(dim, log_lb, log_ub, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    
    # Ensure that bounds are valid (finite numbers)
    if not (np.all(np.isfinite(log_lb)) and np.all(np.isfinite(log_ub))):
        raise ValueError("Bounds must be finite numbers")
    
    # Ensure bounds have proper order: log_lb should be less than or equal to log_ub
    if np.any(log_lb > log_ub):
        raise ValueError("Lower bounds must be less than or equal to upper bounds")
    
    flag = True
    while flag:
        # Generate random uniform numbers for each dimension
        uni = rng.random(dim)
        
        # Calculate log_initpoints within the bounds
        log_initpoints = log_lb + (log_ub - log_lb) * uni
        
        # Check if the new points are within the bounds (they should be)
        if np.all(log_initpoints >= log_lb) and np.all(log_initpoints <= log_ub):
            flag = False
    
    # Return the initial points, exponentiated back from log scale to original scale
    return 10 ** log_initpoints

def vfsa_reannealing(best_cost, best_params, curr_cost, dim, x0, tmax, tscat, data, c, temp_gen, temp0_gen, objective_function):
    log_orig_best_params = np.log10(best_params)
    log_par_delta = log_orig_best_params + 0.01 * log_orig_best_params
    par_delta = 10 ** log_par_delta
    
    cost_delta = np.array([
        objective_function(par_delta if i == j else best_params)
        for j in range(dim)
    ])
    
    par_diff = np.clip(par_delta - best_params, 1e-10, None)  # Avoid division by zero
    s = np.abs((cost_delta - best_cost) / par_diff) * (best_params / best_cost)
    smax = np.max(s)
    
    temp_gen = np.clip(temp_gen * (smax / np.clip(s, 1e-10, None)), 1e-10, None)
    
    k_gen = (-1/c * np.log(np.clip(temp_gen / temp0_gen, 1e-10, None))) ** dim
    k_gen = np.clip(k_gen, 0, None)  # Ensure non-negative values
    
    temp0_acc = curr_cost
    temp_acc = best_cost
    k_acc = (-1/c * np.log(np.clip(temp_acc / temp0_acc, 1e-10, None))) ** dim
    
    return temp_gen, k_gen, temp0_acc, temp_acc, k_acc

def vfsa_temp(temp_gen0, c, k_gen, dim, min_temp=1e-10):
    exponent = -c * np.power(k_gen, 1/dim)
    temp = temp_gen0 * np.exp(exponent)
    return np.clip(temp, min_temp, None)

def simulated_annealing(objective_function, initial_solution, lower_bounds, upper_bounds, 
                        initial_temperature, cooling_rate, max_iterations, neighborhood_function, 
                        log_lb, log_ub, temp_gen, 
                        M=10, eps=0.0, min_temp=1e-10, verbose=False):
    
    current_solution = initial_solution
    current_cost = objective_function(current_solution)
    best_solution = current_solution
    best_cost = current_cost
    temperature = initial_temperature

    reanneal_cost_vec = [best_cost]
    diff = deque(maxlen=M)
    best_diff = deque(maxlen=M)
    count_acc_points = 0

    for iteration in range(max_iterations):
        # Generate a new solution in the neighborhood
        new_solution = neighborhood_function(current_solution, len(current_solution), log_lb, log_ub, temp_gen)
        new_cost = objective_function(new_solution)
        delta_cost = new_cost - current_cost
        
        # Acceptance probability
        if delta_cost < 0 or np.random.rand() < np.exp(-delta_cost / temperature):
            current_solution = new_solution
            current_cost = new_cost
            count_acc_points += 1
            
            # Update the best solution found
            if current_cost < best_cost:
                best_solution = current_solution
                best_cost = current_cost

            # Reannealing: Store the best cost at this point
            reanneal_cost_vec.append(best_cost)
            if len(reanneal_cost_vec) > 1:
                diff.append(abs(reanneal_cost_vec[-1] - reanneal_cost_vec[-2]))
                best_diff.append(abs(reanneal_cost_vec[-1] - best_cost))

                # Termination condition based on the last M accepted costs
                if len(diff) == M and all(d <= eps for d in diff) and all(bd <= eps for bd in best_diff):
                    print('ASA converged, terminating at', iteration)
                    break
        
        # Decrease the temperature with a check for minimum temperature
        temperature = max(min_temp, temperature * cooling_rate)
        
        # Progress print statements
        if iteration % (max_iterations // 10) == 0:
            print(f"Iteration {iteration}")
            if verbose==True:
                print(f"Current cost = {current_cost}, Best cost = {best_cost}")
        
    return best_solution, best_cost

def ASA(objective_function, x0, bounds, maxiter=100, initial_temp=1.0, cooling_rate=0.95, 
        neighborhood_function=vfsa_gen_params, init_function=vfsa_generinitpoint, 
        verbose=False, **kwargs):
    
    dim = len(x0)
    lower_bounds, upper_bounds = np.array(bounds).T
    log_lb, log_ub = np.log10(lower_bounds), np.log10(upper_bounds)
    log_bounds = list(zip(log_lb, log_ub))
    
    # Generate the initial solution
    initial_solution = init_function(dim, log_lb, log_ub)
    
    # Initialize TimeManager
    time_manager = TimeManager()
    
    # Run ASA optimization
    best_solution, best_cost = simulated_annealing(
        objective_function, initial_solution, lower_bounds, upper_bounds,
        initial_temp, cooling_rate, maxiter, neighborhood_function,
        log_lb, log_ub, initial_temp, verbose=False,
        **kwargs
    )
    
    if verbose:
        elapsed_time = time_manager.get_elapsed_time()
        print(f'Starting cost: {best_cost}, Elapsed time for ASA: {elapsed_time:.2f} seconds')
    
    print('ASA best cost:', best_cost)
    print(best_solution)

    # After simulated annealing, refine the solution with a local optimizer (e.g., L-BFGS-B)
    minimizer_kwargs = {
        'method': 'L-BFGS-B',
        'bounds': bounds,
        'options': {
            'disp': True,
            'maxiter': 250,
        }
    }

    local_result = minimize(
        objective_function,
        x0=best_solution,
        **minimizer_kwargs
    )
    
    final_solution = local_result.x
    final_cost = local_result.fun
    nfev = local_result.nfev
    success = local_result.success

    if verbose:
        elapsed_time = time_manager.get_elapsed_time()
        print(f'Final cost: {final_cost}, Elapsed time for polish: {elapsed_time:.2f} seconds')

    return OptimizeResult(x=final_solution, fun=final_cost, nfev=nfev, success=success)

## Wraps Julia ODEs for Python ##

class JuliaODESolution:
    def __init__(self, t, y):
        self.t = np.array(t, dtype=np.float64)  # Time points
        self.y = np.array(y, dtype=np.float64)  # Matrix of solution values

    def __repr__(self):
        return f"JuliaODESolution(t={self.t}, y={self.y})"

def solve_with_julia(t_span, y0, params, param_keys, reinfection=False):
    y0 = np.asarray(y0, dtype=np.float64)

    # Convert params into proper arrays
    if isinstance(params, dict):
        params_julia = np.asarray([params[key] for key in param_keys], dtype=np.float64)
    else:
        params_julia = [np.asarray([p[key] for key in param_keys], dtype=np.float64)
                        for p in params]
    
    # Call Julia model function.
    if reinfection:
        result = Main.tmap_ReinfectionModel_partial(t_span, y0.tolist(), params_julia)
    else:
        result = Main.tmap_LCTModel(t_span, y0.tolist(), params_julia)
    
    # Normalize the result: always an array (list) of tuples.
    if isinstance(result, list):
        sols = [JuliaODESolution(np.array(sol[0], dtype=np.float64),
                                  np.array(sol[1], dtype=np.float64))
                for sol in result]
    else:
        # If not a list, wrap it in a list.
        sols = [JuliaODESolution(np.array(result[0], dtype=np.float64),
                                  np.array(result[1], dtype=np.float64))]

    # If only one solution, optionally return the single object.
    if len(sols) == 1:
        return sols[0]
    else:
        return sols

def JuliaSolve(task):
    def inner_solve(param_set, states, t_span, reinfection=False):
        # Fixed parameter order
        param_order = ["beta", "k", "p", "c", "delta", "xi", "a",
                       "d_E", "delta_E", "K_delta_E", "zeta", "eta", "K_I1", "tau_memory", "damp"]
        
        # Check for vectorized mode: if any parameter’s .val is an array.
        is_vectorized = any(
            isinstance(getattr(param_set, name).val, np.ndarray) and getattr(param_set, name).val.ndim > 0
            for name in param_order if hasattr(param_set, name)
        )
        
        if not is_vectorized:
            params = {name: getattr(param_set, name).val for name in param_order}
            y0_local = States(states).y0
            y0_local[0] = param_set.T0.val
            y0_local[1] = param_set.I10.val
            y0_local[6] = param_set.MP0.val
            sol = solve_with_julia(t_span, y0_local, params, param_keys=param_order, reinfection=reinfection)
            sol.y[4] += param_set.E0.val
            sol.y[5] += param_set.M0.val
            return sol
        else:
            # Build a batch of complete parameter dictionaries.
            n_samples = getattr(param_set, param_order[0]).val.shape[0]
            batch = []
            for i in range(n_samples):
                sample_params = {}
                for name in param_order:
                    val = getattr(param_set, name).val
                    sample_params[name] = val[i] if isinstance(val, np.ndarray) and val.ndim > 0 else val
                batch.append(sample_params)
            y0_local = States(states).y0.copy()
            y0_local[0] = param_set.T0.val
            y0_local[1] = param_set.I10.val
            y0_local[6] = param_set.MP0.val
            sols = solve_with_julia(t_span, y0_local, batch, param_keys=param_order, reinfection=reinfection)
            def get_sample(val, i):
                return val[i] if isinstance(val, np.ndarray) and val.ndim > 0 else val
            for i, sol in enumerate(sols):
                sol.y[4] += get_sample(getattr(param_set, "E0").val, i)
                sol.y[6] += get_sample(getattr(param_set, "M0").val, i)
            return sols
    return inner_solve(*task)

# Curent development area 

class Parameter:
    def __init__(self, name, val=None, bounds=None, method='fixed', space='log10'):
        self.name = name
        self.val = val
        self.method = method
        self.space = space
        if method != 'fixed' and bounds is not None:
            self.l_lim, self.u_lim = self._transform_space(bounds)
        else:
            self.l_lim, self.u_lim = None, None

    def _transform_space(self, bounds):
        lower, upper = bounds
        if self.space == 'log10':
            return np.log10(lower), np.log10(upper)
        elif self.space == 'normal':
            return lower, upper
        return bounds  # Default case

    def _inverse_transform_space(self, value):
        if self.space == 'log10':
            return 10 ** value
        return value

    def __repr__(self):
        return f"{self.val:.2e}"

class Parameters:
    def __init__(self, **kwargs):
        self._parameters = kwargs

    # Provide custom pickling state so that deepcopy doesn't trigger __getattr__ recursively.
    def __getstate__(self):
        return self._parameters

    def __setstate__(self, state):
        self._parameters = state

    def __getattr__(self, item):
        # This only gets called if the attribute isn't found via normal means.
        if item in self._parameters:
            return self._parameters[item]
        raise AttributeError(f"'Parameters' object has no attribute '{item}'")

    def __setattr__(self, key, value):
        if key == '_parameters':
            super().__setattr__(key, value)
        else:
            self._parameters[key] = value

    def __repr__(self):
        return f"Parameters({', '.join([f'{k}={v}' for k, v in self._parameters.items()])})"

    def items(self):
        return self._parameters.items()

    def load_parameters_from_dataframe(self, df_params, patient_id):
        if patient_id not in df_params['id'].values:
            print(f"ID {patient_id} not found in parameter file.")
            return
        param_values = df_params[df_params['id'] == patient_id].iloc[0]
        for param_name, param_obj in self._parameters.items():
            if param_obj.method == 'file' and param_name in param_values:
                param_val = param_values[param_name]
                if not np.isnan(param_val):
                    param_obj.val = param_val
                else:
                    print(f"Parameter '{param_name}' for ID {patient_id} is missing in the parameter file. Falling back to original values.")

class State:
    def __init__(self, label: str, initial_value: float = 0.0, sse: bool = False):
        self.label = label
        self.initial_value = initial_value
        self.sse = sse

    def __repr__(self):
        return f"State(label='{self.label}', initial_value={self.initial_value}, sse={self.sse})"

class States:
    def __init__(self, states_config: list): # Create an ordered list of states 
        self.states = [State(**config) for config in states_config]
        self._state_dict = {state.label: state for state in self.states}

    @property
    def y0(self):
        return np.array([state.initial_value for state in self.states])

    def __getitem__(self, key): # Allows access by index or label
        if isinstance(key, int):
            return self.states[key]
        elif isinstance(key, str):
            try:
                return self._state_dict[key]
            except KeyError:
                raise KeyError(f"State with label '{key}' not found.")
        else:
            raise TypeError("Key must be an integer index or a string label.")

    def __repr__(self):
        return f"States({', '.join(repr(state) for state in self.states)})"
    
class Patient:
    def __init__(self, id, color, t_span, df, parameters, states, df_params=None, sol=None, solve_time=np.nan, sse=np.inf, sse_statewise=np.inf, reinfection=False):
        self.id = id
        self.color = color
        self.t_span = t_span
        self.df = df
        self.parameters = parameters
        self.states = states
        self.sol = sol
        self.solve_time = solve_time
        self.sse = sse
        self.sse_statewise = sse_statewise
        self.reinfection = reinfection

        if df_params is not None:
            self.parameters.load_parameters_from_dataframe(df_params, self.id)
        self.param_names = list(self.parameters._parameters.keys())
        self.results_in_memory = []  # To store results for later DB writes

    def solve(self, verbose=False):
        timer = TimeManager()  # Start timing
        try:
            solutions = JuliaSolve((self.parameters, self.states, self.t_span, self.reinfection))
        except Exception as e:
            solutions = None
            print(f"Error solving patient {self.id}: {e}")
        finally:
            self.solve_time = timer.get_elapsed_time()  # Set elapsed time
            #if verbose: print(f"ID {self.id} solve time: {self.solve_time}")
            del timer  # Destroy the TimeManager instance
            return solutions

    def _compute_sse_for_solution(self, sol, df, states_to_sse):
        """Compute the total SSE and state‐wise SSE for one solution."""
        total_sse = 0
        sse_array = [0] * len(self.states)
        for idx, state in enumerate(self.states):
            state_label = state['label']
            if state_label in states_to_sse and state_label in df.columns:
                data_values = df[state_label].values
                time_points = df['DAY'].values
                valid_indices = ~np.isnan(data_values)
                data_values = data_values[valid_indices]
                time_points = time_points[valid_indices]

                if len(data_values) > 0:
                    model_time_points = sol.t
                    model_values = sol.y[idx]
                    if model_values.shape[0] != len(model_time_points):
                        model_values = np.transpose(model_values)
                    # Interpolate model values to data time points
                    interpolated_model_values = np.interp(time_points, model_time_points, model_values)
                    state_sse = 0
                    for d_val, m_val, t in zip(data_values, interpolated_model_values, time_points):
                        log_diff = (np.log10(max(d_val, 1.0)) - np.log10(max(m_val, 1.0))) ** 2
                        if state_label in ['CD8TE', 'CD8TM'] and t == 0:
                            log_diff *= 10  # Weight for Time = 0, for Population fitting
                        state_sse += log_diff
                    sse_array[idx] = state_sse
                    if state.get('sse', True):
                        total_sse += state_sse
        return total_sse, sse_array

    def compare(self, verbose=False):
        df = self.df
        shedders = [103, 107, 110, 111, 112, 204, 207, 302, 307, 308, 311, 312] + [i for i in range(1, 45) if i not in {10, 33, 34, 35, 37, 38, 39, 40, 41, 42, 43, 44}]
        targets = [self.id]

        # States to include in SSE computation
        states_to_sse = ['V', 'CD8TE', 'CD8TM']

        for pid in targets:
            if isinstance(pid, str) and pid == 'Population':
                pid_df = df[df['VOLUNTEER'].isin(shedders)]
                pid_df['VOLUNTEER'] = 'Population'
            elif isinstance(pid, str) and pid == 'Murine':
                murine_ids = list(range(10))
                pid_df = df[df['VOLUNTEER'].isin(murine_ids)]
                pid_df['VOLUNTEER'] = 'Murine'
            else:
                pid_df = df[df['VOLUNTEER'].isin([pid])]

            # Initialize SSE array
            sse_array = [0] * len(self.states)
            total_sse = 0

            for idx, state in enumerate(self.states):
                state_label = state['label']
                if state_label in states_to_sse and state_label in pid_df.columns:
                    data_values = pid_df[state_label].values
                    time_points = pid_df['DAY'].values
                    valid_indices = ~np.isnan(data_values)
                    data_values = data_values[valid_indices]
                    time_points = time_points[valid_indices]

                    if len(data_values) > 0:
                        model_time_points = self.sol.t
                        model_values = self.sol.y[idx]

                        if model_values.shape[0] != len(model_time_points):
                            model_values = np.transpose(model_values)

                        # Interpolate model values to data time points
                        interpolated_model_values = np.interp(time_points, model_time_points, model_values)

                        # Calculate SSE for this state
                        state_sse = 0
                        for data_val, model_val, time in zip(data_values, interpolated_model_values, time_points):
                            log_diff = (np.log10(max(data_val, 1.0)) - np.log10(max(model_val, 1.0))) ** 2
                            if state_label in ['CD8TE', 'CD8TM'] and time == 0:
                                    log_diff *= 10  # Apply weight of 10 for Time = 0 and CD8TE, CD8TM 
                            state_sse += log_diff

                        sse_array[idx] = state_sse

                        # Accumulate total SSE if the state contributes to SSE
                        if state.get('sse', True):
                            total_sse += state_sse

            # Store total SSE and state-wise SSE
            self.sse = total_sse
            self.sse_statewise = sse_array

            # Extract SSE for specific columns
            state_indices = {state['label']: idx for idx, state in enumerate(self.states)}
            sse_db_components = {label: sse_array[state_indices[label]] for label in states_to_sse if label in state_indices}
            sse_db = sum(sse_db_components.values())

            # Store parameters and SSEs in memory
            all_params = [param.val for param in self.parameters._parameters.values()]
            self.results_in_memory.append((all_params, list(sse_db_components.values()) + [sse_db], str(pid)))

    # TODO fix vectorized (updating results, writing out)
    def objective_function(self, x, verbose=False):
        # Update parameters from the optimizer's input.
        for i, name in enumerate(self.param_names):
            if self.parameters._parameters[name].space == 'log10':
                self.parameters._parameters[name].val = 10 ** x[i]
            else:
                self.parameters._parameters[name].val = x[i]
        # Solve the ODE with the new parameters.
        solutions = self.solve(verbose=verbose)
       
        # Compare the solution(s) with the data.
        if isinstance(solutions, list):
            sses = []
            for solution in solutions:
                self.sol = solution
                self.compare(verbose=verbose)
                sses.append(self.sse)
            # Return the SSE as a list for vectorized mode
            return sses
        else:
            # Handle the case where solutions is a single object
            self.sol = solutions
            self.compare(verbose=verbose)
            # Return the SSE as a scalar for single-set mode
            return self.sse

    def write_results_to_db(self, path):
        conn = sqlite3.connect(path)
        cursor = conn.cursor()

        # Ensure all NumPy arrays are converted properly
        def to_tuple_safe(array):
            """Ensure NumPy arrays are C-contiguous and converted to tuples."""
            if isinstance(array, np.ndarray):
                return tuple(np.ascontiguousarray(array).flatten())  # Flatten to ensure 1D tuple
            return tuple(array)  # Handle standard Python lists/tuples
        
        # Prepare data for insertion
        data_to_insert = [
            to_tuple_safe(p) + to_tuple_safe(e) + (pid,) 
            for p, e, pid in self.results_in_memory
        ]

        cursor.executemany('''
            INSERT INTO evaluations (
                E0, M0, MP0, T0, I10, beta, k, p, c, delta, xi, a, d_E, delta_E, K_delta_E,
                zeta, eta, K_I1, tau_memory, damp, V_sse, CD8TE_sse, CD8TM_sse, sse, PID
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', data_to_insert)

        conn.commit()
        conn.close()
        self.results_in_memory = []  # Flush the results from memory
        print(f'Results saved to {path}.', flush=True)

    def optimize_parameters(self, method='halton', iter=1000, verbose=False, path='output', local_iter=2500, local_method='Nelder-Mead', buff=True, vectorized=False):
        fit_parameters = {name: param for name, param in self.parameters._parameters.items() if param.method in ['fit', 'refine']}

        if not fit_parameters:
            print("No parameters to optimize")
            self.sol = self.solve(verbose=verbose)
            self.compare(verbose=verbose)
            print(f'ID: {self.id} solved in: {self.solve_time}, cost: {self.sse}')
            return (None, self)

        initial_values = []
        for param in fit_parameters.values():
            if param.method == 'fit':
                sampled_value = np.random.uniform(param.l_lim, param.u_lim)
                initial_values.append(sampled_value)
            elif param.method == 'refine':
                initial_values.append(param.val)

        bounds = [(param.l_lim, param.u_lim) for param in fit_parameters.values()]
        self.param_names = list(fit_parameters.keys()) 
        number_fit = len(self.param_names)

        minimizer_kwargs = {
            'method': local_method,
            'bounds': bounds,
            'options': {
                    'xatol': 1e-6,
                    'fatol': 1e-6,
                    'disp': verbose,
                    'maxiter': local_iter,
            }}
        if method == 'differential_evolution':
            print(f'{self.id} Differential Evolution with {iter} generations.',flush=True)
            result = differential_evolution(
                self.objective_function,
                bounds=bounds,
                args=(verbose,),  
                strategy='best1bin',
                maxiter=iter,
                popsize=100,
                mutation=(0.7, 1.9),
                recombination=0.7,
                disp=verbose,
                polish=False,
                init='halton',
                workers=1, 
                vectorized=vectorized,
                updating='deferred' 
            )
        elif method == 'dual_annealing':
            print(f'{self.id} Dual Annealing with {iter} iterations.', flush=True)
            result = dual_annealing(
                self.objective_function,
                bounds=bounds,
                args=(verbose,),
                x0=initial_values,
                no_local_search=False
            )
        elif method == 'direct':
            print(f'{self.id} Direct method optimization.', flush=True)
            result = direct(
                self.objective_function,
                bounds=bounds,
                args=(verbose,), 
                eps=1E-7, 
                maxfun=None,  
                maxiter=iter,  
                locally_biased=False,  
                f_min=-np.inf,  
                f_min_rtol=1E-9,  
                vol_tol=1e-16,  
                len_tol=1e-06, 
                callback=None 
            )
        elif method == 'brute':
            print(f'{self.id} Brute force optimization.', flush=True)
            result = brute(
                self.objective_function,
                ranges=bounds,
                args=(verbose,),
                full_output=True,
                finish=None,  
                workers=1,
                disp=verbose
            )
        elif method == 'shgo':
            print(f'{self.id} SHGO optimization.', flush=True)
            result = shgo(
                self.objective_function,
                bounds=bounds,
                args=(verbose,),
                constraints=None,
                n=100, 
                iters=1,  
                sampling_method='sobol', 
                options={
                    'disp': verbose,
                    'maxiter': iter,  
                    'ftol': 1e-6, 
                    'xatol': 1e-6,  
                    'fatol': 1e-6,  
                }
            )
        elif method == 'ASA':
            result = ASA(
                self.objective_function,
                x0=initial_values, 
                bounds=bounds,
                args=(verbose,),
                maxiter=iter,
                initial_temp=1.0,
                cooling_rate=0.95,
                neighborhood_function=vfsa_gen_params,
                init_function=vfsa_generinitpoint,
                M=15,  # Number of accepted costs for convergence
                eps=1e-6
            )
        elif method == 'basin_hopping':
            class BoundedStep:
                def __init__(self, fit_parameters, stepsize=0.5, normal_sampling=False):
                    self.fit_parameters = fit_parameters
                    self.stepsize = stepsize
                    self.normal_sampling = normal_sampling
                    self.precomputed = {}

                    for i, (param_name, param) in enumerate(self.fit_parameters.items()):
                        lower, upper = param.l_lim, param.u_lim
                        if param.space == 'log10':
                            lower, upper = np.log10(lower), np.log10(upper)

                        self.precomputed[i] = {
                            'lower': lower,
                            'upper': upper,
                            'range': upper - lower,
                            'is_log': param.space == 'log10'
                        }

                def __call__(self, x):
                    rng = np.random.default_rng()

                    for i in range(len(x)):
                        bounds = self.precomputed[i]
                        lower, upper, range_, is_log = bounds['lower'], bounds['upper'], bounds['range'], bounds['is_log']

                        if self.normal_sampling:
                            step = rng.normal(0, self.stepsize)
                            step = np.clip(step, -3, 3)
                            normalized_step = (step + 3) / 6
                        else:
                            normalized_step = rng.uniform()

                        step = (normalized_step - 0.5) * self.stepsize * range_
                        x[i] = np.clip(x[i] + step, lower, upper)

                        if is_log:
                            x[i] = 10 ** x[i]

                    return x

            print(f'{self.id} Basin Hopping with n = {iter}',flush=True)
            result = basinhopping(
                self.objective_function,  
                args=(verbose,),
                x0=initial_values,
                niter=iter,
                T=0.01,
                stepsize=1.0,
                minimizer_kwargs=minimizer_kwargs,
                take_step=BoundedStep(fit_parameters, stepsize=1.0),
                interval=1,
                disp=verbose,
                niter_success=2,
                target_accept_rate=0.5,
                stepwise_factor=0.8,
            )
        elif method == 'halton':
            print(f'{self.id} Halton with n = {iter}', flush=True)
            # Halton sampling-based optimization
            sampler = qmc.Halton(d=number_fit)
            n_samples = iter  # Number of samples for Halton sequence
            step_size = n_samples // 20  # 5% of n_samples

            # Generate Halton samples in [0, 1] and scale them to the parameter bounds
            halton_samples = sampler.random(n_samples)
            scaled_samples = qmc.scale(halton_samples, [b[0] for b in bounds], [b[1] for b in bounds])
            best_sample = None
            best_sse = np.inf
            progress_count = 0

            for i, sample in enumerate(scaled_samples):
                sse = self.objective_function(sample)
                if sse < best_sse:
                    best_sse = sse
                    best_sample = sample

                # Print progress every 5% 
                if (i + 1) % step_size == 0:
                    progress_count += 5
                    print(f"Progress: {progress_count}% of samples evaluated ({i+1}/{n_samples})")

            # Assign the best-found parameters back to the model
            result = OptimizeResult()
            result.x = best_sample
            result.fun = best_sse

        print(result.x)
        if buff == True:
            # Local solver "buff" (polish, but that keyword is used in some global algos already)
            print(f'{self.id} polish', flush=True)
            buff_method = 'L-BFGS-B'

            # Define all possible options
            options = {
                'disp': False,
                'maxiter': 1E4,
                'gtol': 1e-9,
                'norm': None,
                'return_all': False,
                'initial_trust_radius': None,
            }

            try:
                method_options = show_options('minimize', buff_method, disp=False)
                valid_keys = set(method_options.keys()) if isinstance(method_options, dict) else set()
            except Exception:
                valid_keys = set()  

            filtered_options = {k: v for k, v in options.items() if k in valid_keys}

            buff_minimizer_kwargs = {
                'method': buff_method,
                'bounds': bounds,
                'options': filtered_options
            }

            # Suppress OptimizeWarning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)

                # Run minimization
                result = minimize(
                self.objective_function, 
                x0=result.x if 'result' in locals() else initial_values,
                **buff_minimizer_kwargs
            )

        #elif not hasattr(result, "x"):
                #result.x = initial_values

        # Write results to database
        db_path = f'../sql/{path}.db'
        os.makedirs('sql', exist_ok=True)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                E0 REAL, M0 REAL, MP0 REAL, T0 REAL, I10 REAL, beta REAL, k REAL,
                p REAL, c REAL, delta REAL, xi REAL, a REAL,
                d_E REAL, delta_E REAL, K_delta_E REAL, zeta REAL, eta REAL, K_I1 REAL, tau_memory REAL, damp REAL,
                V_sse REAL, CD8TE_sse REAL, CD8TM_sse REAL, sse REAL, PID REAL
            )
        ''')
        conn.commit()
        conn.close()
        self.write_results_to_db(path=db_path)
        
        # Final update of parameters using the Parameter's inverse transformation.
        new_x = []
        for i, name in enumerate(self.param_names):
            param_obj = self.parameters._parameters[name]
            new_val = param_obj._inverse_transform_space(result.x[i])
            param_obj.val = new_val
            new_x.append(new_val)
            print(f'Parameter {name}: {new_val}')
        
        # Overwrite result.x so that it is now in normal space.
        result.x = np.array(new_x)
        
        # Solve and compare with the best-found parameters.
        self.sol = self.solve(verbose=verbose)
        self.results_in_memory = []
        
        return (result, self)

    def __repr__(self):
        return f"Patient({self.id}, sse={self.sse}, parameters={self.parameters})"

class Patients:
    def __init__(self, ids, df, t_span, parameters, states, parameter_file=None, reinfection=False, colors='unique'):
        if colors == 'unique':
            self.color_dict = self.assign_named_colors(ids)
        elif colors == 'black':
            self.color_dict = self.assign_black_colors(ids)
        else:
            raise ValueError("Invalid color mode. Choose 'unique' or 'black'.")
        t_fill = np.linspace(t_span[0], t_span[-1], 200)
        t_int = np.arange(t_span[0], t_span[-1] + 1)
        self.t_span = np.unique(np.concatenate([t_span, t_fill, t_int]))
        self.df = df
        self.parameters = parameters
        self.states = states
        self.reinfection = reinfection
        # Load the parameters from the Excel file once
        if parameter_file:
            self.df_params = read_excel(os.path.abspath(parameter_file))
        else:
            print('No parameters loaded: No file specified. Add parameter_file= to Patients initialization call.')
            self.df_params = None

        self.patients = {
            id: Patient(
                id,
                self.color_dict[id],
                self.t_span,
                self.df,
                copy.deepcopy(self.parameters),
                copy.deepcopy(self.states),
                self.df_params,
                reinfection=self.reinfection,
            ) for id in ids
        }

        self.parameters.patients = self.patients

    def optimize_parameters(self, opt_target, method, iter, path, verbose=False, local_iter=1000, local_method='L-BFGS-B', buff=True, vectorized=False):
        results = []
        for patient in self.patients.values():
            if patient.id == opt_target:
                result = patient.optimize_parameters(
                    method=method, iter=iter, verbose=verbose, path=path, local_iter=local_iter, local_method=local_method, buff=buff, vectorized=vectorized)
                results.append(result)
        return results

    def assign_named_colors(self, ids):
        """Assigns a repeating cycle of named colors converted to RGB tuples."""
        named_colors = [
            'gray', 'purple', 'magenta', 'red', 'goldenrod', 'darkorange',
            'saddlebrown', 'mediumblue', 'dodgerblue', 'turquoise', 'darkgreen', 'lawngreen'
        ]
        rgb_colors = sns.color_palette(named_colors)  # Convert to RGB tuples
        num_colors = len(named_colors)
        return {id_: rgb_colors[i % num_colors] for i, id_ in enumerate(ids)}

    def assign_black(self, ids):
        """Assigns solid black to all patients."""
        return {id_: (0, 0, 0) for id_ in ids}
    
    def __getitem__(self, id):
        patient = self.patients.get(id, None)
        if patient is not None:
            return repr(patient)
        return None

    def __repr__(self):
        return '\n'.join(repr(patient) for patient in self.patients.values())

