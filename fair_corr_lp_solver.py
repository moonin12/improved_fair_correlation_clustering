'''
Excerpt from codes by Suman K. Bera in https://github.com/nicolasjulioflores/fair_algorithms_for_clustering
'''
import logging
import time

from cplex import Cplex


def get_edge_name(u, v):
    if u < v:
        return "x_" + str(u) + "_" + str(v)
    elif u > v:
        return "x_" + str(v) + "_" + str(u)
    return None


def get_variables(graph):
    logging.debug("get_variables --------------")
    nr_nodes = graph["nr_nodes"]
    # Variable names are in format x_u_v where u < v
    names = ["x_{}_{}".format(j, i) for i in range(nr_nodes) for j in range(i)]
    logging.debug(names)
    nr_variables = len(names)
    # Objective coefficients are -1 and 1 for negative and positive edges respectively
    # Initially, set all objective coefficients to be -1
    objective_coeff = dict.fromkeys(names, -1)
    for edge in graph["positive_edges"]:
        logging.debug(edge)
        # WARNING: We assume edges are saved in the format [u,v] u < v. if the assumption changes, use get_edge_name(u, v)
        edge_var_name = "x_" + str(edge[0]) + "_" + str(edge[1])
        objective_coeff[edge_var_name] = 1

    logging.debug("vars in obj -------")
    logging.debug(objective_coeff)

    logging.debug("Nr variables " + str(nr_variables) + " object coeff len " + str(len(objective_coeff.keys())))
    variables = {"obj": list(objective_coeff.values()),
                 "lb": [0] * nr_variables,
                 "ub": [1] * nr_variables,
                 "names": names}
    return variables


def get_constraints(graph, alphas):
    nr_nodes = graph["nr_nodes"]
    rows = []
    rhs = []
    senses = []
    names = []

    def add_tri_ineq_constraints():
        logging.debug("get_tri_ineq --------------")
        nonlocal rows, rhs, senses, names
        # Constraints x_u_v + x_v_w >= x_u_w for u < v < w  (take x_u_w to the other side)
        new_constraints = [
            [["x_{}_{}".format(u, v), "x_{}_{}".format(v, w), "x_{}_{}".format(u, w)], [1, 1, -1]]
            for w in range(nr_nodes) for v in range(w) for u in range(v)]
        new_names = [
            "tri_ineq_{}_{}_{}".format(u, v, w)
            for w in range(nr_nodes) for v in range(w) for u in range(v)]
        logging.debug(new_constraints)
        nr_new_constraints = len(new_constraints)

        logging.debug("nr_new_constraints " + str(nr_new_constraints) + " nr constraint names " + str(len(new_names)))
        rows.extend(new_constraints)
        rhs.extend([0] * nr_new_constraints)
        senses.extend(["G"] * nr_new_constraints)
        names.extend(new_names)

    def add_fairness_constraints():
        logging.debug("get_fairness --------------")
        nonlocal rows, rhs, senses, names

        nodes_by_color = graph["nodes_by_color"]
        color_dist = graph["color_dist"]
        min_nr_color = graph["min_nr_color"]
        max_nr_color = graph["max_nr_color"]
        nr_colors = max_nr_color - min_nr_color + 1

        # Constraints sum_{v \in V_i} (1-x_u_v) <= \alpha_i [sum_{v \in V} (1-x_u_v)] for all colors i and nodes u
        # Rearrange:  \alpha_i sum_{v \in V} x_u_v -  sum_{v \in V_i} x_u_v <=  \alpha_i |V| - |V_i|
        senses.extend(["L"] * (nr_nodes * nr_colors))
        for i in range(min_nr_color, max_nr_color + 1):  # for each color i
            # rhs is the same regardless of what u is
            row_i_rhs = alphas[i] * nr_nodes - color_dist[i]
            rhs.extend([row_i_rhs] * nr_nodes)
            names.extend(["fair_{}_{}".format(i, u) for u in range(nr_nodes)])
            nodes_color_i = nodes_by_color[i]
            logging.debug("nodes of color i ---")
            logging.debug(nodes_color_i)
            for u in range(nr_nodes):
                # We'll be using all variables x_v_u for v < u and x_u_v for v > u
                # Note: x_u_u = 0 so can be ignored
                var_names = ["x_{}_{}".format(v, u) for v in range(u)]
                var_names.extend(["x_{}_{}".format(u, v) for v in range(u + 1, nr_nodes)])

                # All such variables appear with \alpha_i coefficient in the lhs
                var_coeff = dict.fromkeys(var_names, alphas[i])

                # If a node is of color i, it has a -1 coefficient IN ADDITION to the \alpha_i
                for node_i in nodes_color_i:
                    edge_var_name = get_edge_name(u, node_i)
                    if edge_var_name is None:
                        continue
                    var_coeff[edge_var_name] -= 1

                logging.debug("row for u " + str(u) + " and i " + str(i) + "-------")
                row_u_i = [list(var_coeff.keys()), list(var_coeff.values())]
                logging.debug(row_u_i)
                rows.append(row_u_i)

    add_tri_ineq_constraints()
    add_fairness_constraints()

    constraints = {"lin_expr": rows, "senses": senses, "rhs": rhs, "names": names}
    return constraints


# Returns the CPLEX problem in addition to variable names, in the same order used in the LP.
def get_fair_corr_lp(graph, alphas):
    if len(alphas.keys()) != len(graph["color_dist"].keys()):
        raise Exception("alphas size is " + str(alphas) + " does not match graph nr_colors " + str(
            len(graph["color_dist"])))

    # Step 1. Initiate a model for cplex.
    print("Initializing Cplex model")
    problem = Cplex()

    # Step 2. Declare that this is a minimization problem
    problem.objective.set_sense(problem.objective.sense.minimize)

    # Step 3.   Declare and  add variables to the model. The function
    print("Starting to add variables...")
    t1 = time.monotonic()
    variables = get_variables(graph)
    problem.variables.add(obj=variables["obj"],
                          lb=variables["lb"],
                          ub=variables["ub"],
                          names=variables["names"])
    t2 = time.monotonic()
    print("Completed. Time for creating and adding variable = {}".format(t2 - t1))

    # Step 4.   Declare and add constraints to the model.
    print("Starting to add constraints...")
    t1 = time.monotonic()
    constraints = get_constraints(graph, alphas)
    problem.linear_constraints.add(lin_expr=constraints["lin_expr"],
                                   senses=constraints["senses"],
                                   rhs=constraints["rhs"],
                                   names=constraints["names"])
    t2 = time.monotonic()
    print("Completed. Time for creating and adding constraints = {}".format(t2 - t1))

    return problem, variables["names"]


def solve_fair_corr_lp(graph, alphas, lp_method=0):
    t1 = time.monotonic()
    problem, variable_names = get_fair_corr_lp(graph, alphas)
    t2 = time.monotonic()
    lp_defining_time = t2 - t1

    t1 = time.monotonic()
    problem.parameters.lpmethod.set(lp_method)
    # problem.parameters.barrier.convergetol.set(0.1) # Somehow increases runtime???
    problem.solve()
    t2 = time.monotonic()
    lp_solving_time = t2 - t1

    nr_nodes = graph["nr_nodes"]
    nr_positive_edges = graph["nr_positive_edges"]
    nr_negative_edges = (nr_nodes * (nr_nodes - 1)) / 2 - nr_positive_edges
    # LP objective is sum_{(u,v) \in E^+} x_{uv} + sum_{(u,v) \in E^-} (1 - x_{uv})
    # = sum_{(u,v) \in E^+} x_{uv} + |E^-| - sum_{(u,v) \in E^-} x_{uv}
    # referring to get_fair_corr_lp, the |E^-| is a constant number so it is not added. We manually add it here.
    cost = problem.solution.get_objective_value() + nr_negative_edges

    raw_values = problem.solution.get_values()  # The values assigned to variables in the LP solution
    values = dict(zip(variable_names, raw_values))
    res = {
        "lp_defining_time": lp_defining_time,
        "lp_solving_time": lp_solving_time,
        "time": lp_solving_time + lp_defining_time,
        "status": problem.solution.get_status(),
        "success": problem.solution.get_status_string(),
        "cost": cost,
        "nr_variables": len(variable_names),
        "values": values,
    }
    return res
