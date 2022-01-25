#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 10:44:00 2022

@author: alejomonbar
"""
import numpy as np
import matplotlib.pyplot as plt
from docplex.mp.model import Model

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_optimization.algorithms import CplexOptimizer

from qiskit import Aer

def BinPacking(num_items, num_bins, weights, max_weight, simplification=False):
    # Construct model using docplex
    mdl = Model("BinPacking")

    y = mdl.binary_var_list(num_bins, name="y") # list of variables that represent the bins
    x =  mdl.binary_var_matrix(num_items, num_bins, "x") # variables that represent the items on the specific bin

    objective = mdl.sum(y)

    mdl.minimize(objective)

    for i in range(num_items):
        # First set of constraints: the items must be in any bin
        mdl.add_constraint(mdl.sum(x[i, j] for j in range(num_bins)) == 1)

    for j in range(num_bins):
        # Second set of constraints: weight constraints
        mdl.add_constraint(mdl.sum(weights[i] * x[i, j] for i in range(num_items)) <= max_weight * y[j])

    # Load quadratic program from docplex model
    qp = QuadraticProgram()
    qp.from_docplex(mdl)
    if simplification:
        l = int(np.ceil(np.sum(weights)/max_weight))
        qp = qp.substitute_variables({f"y_{_}":1 for _ in range(l)}) # First simplification 
        qp = qp.substitute_variables({"x_0_0":1}) # Assign the first item into the first bin
        qp = qp.substitute_variables({f"x_0_{_}":0 for _ in range(1, num_bins)}) # as the first item is in the first 
                                                                                #bin it couldn't be in the other bins
    qubo = QuadraticProgramToQubo().convert(qp)# Create a converter from quadratic program to qubo representation
    return qubo, qp

def BinPackingNewApproach(num_items, num_bins, weights, max_weight, alpha=0.01, simplification=False):
    # Construct model using docplex
    mdl = Model("BinPackingNewApproach")

    y = mdl.binary_var_list(num_bins, name="y") # list of variables that represent the bins
    x =  mdl.binary_var_matrix(num_items, num_bins, "x") # variables that represent the items on the specific bin

    objective = mdl.sum(y)
    
    # PENALIZATION
    penalization = 0
    for j in range(num_bins):
        t = max_weight * y[j] - mdl.sum(weights[i] * x[i, j] for i in range(num_items))
        penalization += 10*(t**2 - t)
    mdl.minimize(objective + alpha * penalization)

    for i in range(num_items):
        # First set of constraints: the items must be in any bin
        mdl.add_constraint(mdl.sum(x[i, j] for j in range(num_bins)) == 1)

    # Load quadratic program from docplex model
    qp = QuadraticProgram()
    qp.from_docplex(mdl)
    if simplification:
        l = int(np.ceil(np.sum(weights)/max_weight))
        qp = qp.substitute_variables({f"y_{_}":1 for _ in range(l)}) # First simplification 
        qp = qp.substitute_variables({"x_0_0":1}) # Assign the first item into the first bin
        qp = qp.substitute_variables({f"x_0_{_}":0 for _ in range(1, num_bins)}) # as the first item is in the first 
                                                                                #bin it couldn't be in the other bins
    qubo = QuadraticProgramToQubo().convert(qp)# Create a converter from quadratic program to qubo representation
    return qubo

def Knapsack(weights, values, max_weight):
    mdl = Model("Knapsack")
    
    num_items = len(weights)
    x = mdl.binary_var_list(num_items, name="x")
    
    objective = mdl.sum([x[i]*values[i] for i in range(num_items)])
    mdl.maximize(objective)
    
    mdl.add_constraint(mdl.sum(weights[i] * x[i] for i in range(num_items)) <= max_weight)
    # Converting to QUBO
    qp = QuadraticProgram()
    qp.from_docplex(mdl)
    qubo = QuadraticProgramToQubo().convert(qp)
    return qubo
    


def interpret(results, weights, max_weight, num_items, num_bins, simplify=False):
    """
    Save the results as a list of list where each sublist represent a bin
    and the sublist elements represent the items weights
    
    Args:
    results: results of the optimization
    weights (list): weights of the items
    max_weight (int): Max weight of a bin
    num_items: (int) number of items
    num_bins: (int) number of bins
    """
    if simplify:
        l = int(np.ceil(np.sum(weights)/max_weight))
        bins = l * [1] + list(results[:num_bins - l])
        items = results[num_bins - l: (num_bins - l) + num_bins * (num_items - 1)].reshape(num_items - 1, num_bins)
        items_in_bins = [[i+1 for i in range(num_items-1) if bins[j] and items[i, j]] for j in range(num_bins)]
        items_in_bins[0].append(0)
    else:
        bins = results[:num_bins]
        items = results[num_bins:(num_bins + 1) * num_items].reshape((num_items, num_bins))
        items_in_bins = [[i for i in range(num_items) if bins[j] and items[i, j]] for j in range(num_bins)]
    return items_in_bins

def get_figure(items_in_bins, weights, max_weight, title=None):
    """Get plot of the solution of the Bin Packing Problem.

    Args:
        result : The calculated result of the problem

    Returns:
        fig: A plot of the solution, where x and y represent the bins and
        sum of the weights respectively.
    """
    colors = plt.cm.get_cmap("jet", len(weights))
    num_bins = len(items_in_bins)
    fig, axes = plt.subplots()
    for _, bin_i in enumerate(items_in_bins):
        sum_items = 0
        for item in bin_i:
            axes.bar(_, weights[item], bottom=sum_items, label=f"Item {item}", color=colors(item))
            sum_items += weights[item]
    axes.hlines(max_weight, -0.5, num_bins - 0.5, linestyle="--", color="tab:red", label="Max Weight")
    axes.set_xticks(np.arange(num_bins))
    axes.set_xlabel("Bin")
    axes.set_ylabel("Weight")
    axes.legend()
    if title:
        axes.set_title(title)
    return fig

def qaoa_circuit(qubo: QuadraticProgram, p: int = 1):
    """
    Given a QUBO instance and the number of layers p, constructs the corresponding parameterized QAOA circuit with p layers.
    Args:
        qubo: The quadratic program instance
        p: The number of layers in the QAOA circuit
    Returns:
        The parameterized QAOA circuit
    """
    size = len(qubo.variables)
    qubo_matrix = qubo.objective.quadratic.to_array(symmetric=True)
    qubo_linearity = qubo.objective.linear.to_array()

    #Prepare the quantum and classical registers
    qaoa_circuit = QuantumCircuit(size,size)
    #Apply the initial layer of Hadamard gates to all qubits
    qaoa_circuit.h(range(size))

    #Create the parameters to be used in the circuit
    gammas = ParameterVector('gamma', p)
    betas = ParameterVector('beta', p)

    #Outer loop to create each layer
    for i in range(p):
        
        #Apply R_Z rotational gates from cost layer
        for j in range(size):
            qaoa_circuit.rz((qubo_linearity[j] + np.sum(qubo_matrix[j]))*gammas[i], j)
        #Apply R_ZZ rotational gates for entangled qubit rotations from cost layer
        for j in range(size-1):
            for k in range(j+1,size):
                qaoa_circuit.cx(k,j)
                qaoa_circuit.rz(qubo_matrix[j,k]*gammas[i], j)
                qaoa_circuit.cx(k,j)
#                     qaoa_circuit.cp(0.5*qubo_matrix[j,k]*gammas[i], j, k)
                        
        # Apply single qubit X - rotations with angle 2*beta_i to all qubits
        qaoa_circuit.rx(2*betas[i],range(size))
    qaoa_circuit.measure(range(size), range(size))
    return qaoa_circuit

def cost_func(parameters, circuit, objective, backend=Aer.get_backend("qasm_simulator")):
    """
    This function returns the cost function Eq. 5 for the circuit created with the function ground_circuit.

    Parameters
    ----------
    params : list.
        List of angles gamma and beta with size equal to the depth.
    G : networkx graph
        Graph with the information of number of vertices, edges and weights.
    depth : int
        number of steps the unitarity is applied.
    shots: int
        Circuit number of repetition of the quantum circuit (To get the statistics)
    imp: Boolean
        Improvement described in section 4.9
    backend: IBMQ backend to solve the problem
    Returns
    -------
    qc : qiskit circuit
        Circuit with a x-rotation of every qubit.
    """
    cost = 0
    counts = backend.run(circuit.assign_parameters(parameters=parameters)).result().get_counts()
    counts = {k: v for k, v in sorted(counts.items(), key=lambda item: item[1], reverse=True)}
    samples = 0
    for sample in list(counts.keys())[:10]:
        cost += counts[sample] * objective.evaluate([int(_) for _ in sample])
        samples += counts[sample]
    return cost / samples


def new_eq_optimal(qubo_new, qubo_classical):
    """
    From the classical solution and considering that cplex solution is the optimal, we can traslate the optimal
    solution to the QUBO representation based on our approach.
    
    
    """
    num_vars = qubo_new.get_num_vars()
    result_cplex = CplexOptimizer().solve(qubo_classical)
    result_new_ideal = qubo_new.objective.evaluate(result_cplex.x[:num_vars])# Replacing the ideal solution into
                                               #our new approach to see the optimal solution on the new objective
                                               #function
    return result_new_ideal

def eval_constrains(qp, result):
    constraints = qp.linear_constraints
    varN = len(qp.variables)
    eval_const = []
    for const in constraints:
        eval_const.append(const.evaluate(result.x[:varN]))
    return np.array(eval_const) 
