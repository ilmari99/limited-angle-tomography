import itertools
import sympy
from sympy.solvers import diophantine
from cache_to_disk import cache_to_disk

@cache_to_disk(30)
def find_convolution_parameters_grid_search(num_layers,
                                            input_side_len,
                                            output_side_len,
                                            fix_kernel_size = None,
                                            fix_padding = None,
                                            fix_stride = None,
                                            transposed_convolutions = False,
                                            grid_update = {},
                                            ):
    """ Given a square input, find square kernels, paddings and strides,
    such that the output is also a square of a desired size.
    """
    grid = {
        "kernel_size" : [2, 3, 5, 7, 9],
        "padding" : [0, 1, 2, 3, 4, 5],
        "stride" : [1, 2, 3, 4, 5]
    }
    grid.update(grid_update)
    
    if fix_kernel_size is not None:
        grid["kernel_size"] = [fix_kernel_size]
    if fix_padding is not None:
        grid["padding"] = [fix_padding]
    if fix_stride is not None:
        grid["stride"] = [fix_stride]
    
    parameters = {f"k_{i}" for i in range(num_layers)}
    parameters.update({f"p_{i}" for i in range(num_layers)})
    parameters.update({f"s_{i}" for i in range(num_layers)})
    print(f"Trying to find parameters for {num_layers} layers, input side length {input_side_len}, output side length {output_side_len}")
    print(f"Grid: {grid}")
    # Find all possible combinations
    kernel_size_gen = itertools.product(grid["kernel_size"], repeat=num_layers)
    padding_size_gen = itertools.product(grid["padding"], repeat=num_layers)
    stride_size_gen = itertools.product(grid["stride"], repeat=num_layers)
    solution = (None, None, None)
    num_tries = 0
    for kernel_size in kernel_size_gen:
        print(f"Trying kernel size: {kernel_size}", end="\r")
        padding_size_gen = itertools.product(grid["padding"], repeat=num_layers)  # Reset padding generator
        for padding in padding_size_gen:
            stride_size_gen = itertools.product(grid["stride"], repeat=num_layers)  # Reset stride generator
            for stride in stride_size_gen:
                num_tries += 1
                #print(f"Trying kernel size: {kernel_size}, padding: {padding}, stride: {stride}")
                if transposed_convolutions:
                    output_size = calculate_output_size(input_side_len, kernel_size, padding, stride, transposed_convolutions=True)
                else:
                    output_size = calculate_output_size(input_side_len, kernel_size, padding, stride)
                if output_size == output_side_len:
                    #print(f"\nFound solution: {kernel_size}, {padding}, {stride}")
                    solution = (kernel_size, padding, stride)
                    break
            if solution[0] is not None:
                break
        if solution[0] is not None:
            break
    print(f"Tried {num_tries} parameter combinations")
    return solution[0], solution[1], solution[2]
    
def calculate_output_size(input_size, kernel_sizes, paddings, strides, transposed_convolutions = False):
    """ Calculate the output size of a convolutional NN
    """
    output_size = input_size
    for i in range(len(kernel_sizes)):
        if transposed_convolutions:
            output_size = (output_size - 1) * strides[i] - 2 * paddings[i] + kernel_sizes[i]
        else:
            output_size = (output_size - kernel_sizes[i] + 2 * paddings[i]) / strides[i] + 1
    return output_size

#@cache_to_disk(30)
def sympy_solve(system_of_equations, solve_for = "s", num_layers=3):
    """ Given a system of equations
    """
    keys_to_solve = []
    for layer_num in range(num_layers):
        if solve_for == "k":
            keys_to_solve.append(sympy.symbols(f"K_{layer_num}h"))
        elif solve_for == "p":
            keys_to_solve.append(sympy.symbols(f"P_{layer_num}h"))
        elif solve_for == "s":
            keys_to_solve.append(sympy.symbols(f"S_{layer_num}h"))
        keys_to_solve.append(sympy.symbols(f"d_{layer_num+1}h"))
            
    solution = sympy.solve(system_of_equations, keys_to_solve, dict=True)
    if isinstance(solution, list):
        solution = solution[0]
    print(f"Solution: {solution}")
    return solution


def find_convolution_parameters(num_layers, input_side_len, output_side_len, solve_for="s", free_kernel_size = 3, free_padding = 0, free_stride = 1, transposed_convolutions = False):
    """ Given a square input, find square kernels, paddings and strides,
    such that the output is also a square of a desired size.
    """
    assert solve_for in ["k", "p", "s"], "Solve must be one of 'k', 'p', 's'"
    system_of_equations = []
    for i in range(num_layers):
        # For transposed convolutions:
        if transposed_convolutions:
            eq = ((sympy.symbols(f"d_{i}h") - 1) * sympy.symbols(f"S_{i}h") - 2 * sympy.symbols(f"P_{i}h") + sympy.symbols(f"K_{i}h"))
        else:
            eq = ((sympy.symbols(f"d_{i}h") - sympy.symbols(f"K_{i}h") + 2 * sympy.symbols(f"P_{i}h")) / sympy.symbols(f"S_{i}h")) + 1
        system_of_equations.append(sympy.Eq(sympy.symbols(f"d_{i+1}h"), eq))
    
    # Add the constraints
    system_of_equations.append(sympy.Eq(sympy.symbols(f"d_0h"), input_side_len))
    system_of_equations.append(sympy.Eq(sympy.symbols(f"d_{num_layers}h"), output_side_len))
    
    # Find an integer solution with Diophatine
    solution = sympy_solve(system_of_equations, solve_for = solve_for)
    
    print(solution)
    all_keys = []
    for layer_num in range(num_layers):
        wh = "h"
        all_keys.append(f"K_{layer_num}{wh}")
        all_keys.append(f"P_{layer_num}{wh}")
        all_keys.append(f"S_{layer_num}{wh}")
        all_keys.append(f"d_{layer_num+1}{wh}")
            
    all_keys = [sympy.symbols(key) for key in all_keys]
    free_variables = set(all_keys) - set(solution.keys())
    print(f"Free variables: {free_variables}")
    # The solution has a mapping from sym -> eq,
    # where the free variables define the solution
    # Let's find the solution where all the free variables are 1
    subs_map = {"K" : free_kernel_size, "P" : free_padding, "S" : free_stride}
    if transposed_convolutions:
        d_map = {f"d_{i}h" : output_side_len // (2**(num_layers - i)) for i in range(0,num_layers+1)}
    else:
        d_map = {f"d_{i}h" : input_side_len // 2**i for i in range(0,num_layers+1)}
    subs_map.update(d_map)
    print(f"Subs map: {subs_map}")
    
    solution = try_subs_map(subs_map, free_variables, solution)
    print(f"Solution: {solution}")
    kernel_sizes = [(solution[f"K_{i}h"], solution[f"K_{i}h"]) for i in range(num_layers)]
    paddings = [(solution[f"P_{i}h"], solution[f"P_{i}h"]) for i in range(num_layers)]
    strides = [(solution[f"S_{i}h"], solution[f"S_{i}h"]) for i in range(num_layers)]
    print(f"To get from {input_side_len} to {output_side_len} (Transp.={transposed_convolutions}), use the following parameters:")
    print(f"Kernel sizes: {kernel_sizes}, Paddings: {paddings}, Strides: {strides}")
    if not all([all([x > 0 for x in kernel_sizes[i]]) for i in range(num_layers)]):
        raise ValueError("Kernel sizes must be positive")
    if not all([all([x >= 0 for x in paddings[i]]) for i in range(num_layers)]):
        raise ValueError("Paddings must be non-negative")
    if not all([all([x > 0 for x in strides[i]]) for i in range(num_layers)]):
        raise ValueError("Strides must be positive")
    return kernel_sizes, paddings, strides

def try_subs_map(subs_map, free_variables, reduced_system):
    reduced_system = reduced_system.copy()
    for free_var in free_variables:
        key1 = str(free_var.name).strip()
        key2 = str(free_var.name[0]).strip()
        value = subs_map.get(key1, subs_map.get(key2, None))
        if value is None:
            raise ValueError(f"Value not found for keys: {key1}, {key2}")
        reduced_system[sympy.sympify(free_var)] = value
    print(f"Reduced system: {reduced_system}")
    # Solve
    system_of_equations = {key: value for key, value in reduced_system.items()}
    solution = diophantine(system_of_equations)
    print(f"Solution: {solution}")
    if isinstance(solution, list):
        solution = solution[0]
    solution = {str(key): value for key, value in solution.items()}
    return solution