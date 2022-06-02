import logging
import random

from fair_corr_lp_solver import get_edge_name
from utils.clustering_utils import eval_clustering


def fair_corr_clustering(graph, lp_res, epsilon, sigma, rho, alphas, do_shuffle=False):
    logging.debug("------- fair_corr_clustering")
    nr_nodes = graph["nr_nodes"]
    max_nr_color = graph["max_nr_color"]
    min_nr_color = graph["min_nr_color"]
    node_colors = graph["colors"]
    x_values = lp_res["values"]

    logging.debug("nr nodes " + str(nr_nodes))
    nondegen_clusters = []
    nondegen_color_dists = []
    # In the beginning, all the nodes are uncovered
    uncovered = set(range(0, nr_nodes))

    while len(uncovered) > 0:
        logging.debug("**** new iteration")
        # made_nondegen_cluster is used to break the loop in case we don't find a u that can be the center of a
        # non-degenerate cluster
        made_nondegen_cluster = False

        def get_T_u(u):
            T_u = [u]
            # sum of x_uv's for v in T_u
            T_u_sum_dist = 0
            # Construct T_u and fill in T_u_sum_dist
            for v in uncovered:
                if v == u:
                    continue
                edge_str = get_edge_name(u, v)
                x_uv = x_values[edge_str]
                if x_uv <= rho:
                    T_u.append(v)
                    T_u_sum_dist += x_uv

            # dictionary of T_u's color distribution, used later to evaluate fairness
            T_u_color_dist = dict.fromkeys(range(min_nr_color, max_nr_color + 1), 0)
            # Fill in T_u_color_dist
            for v in T_u:
                # Colors that v has
                v_color_list = node_colors[v]
                for color in v_color_list:
                    T_u_color_dist[color] += 1

            return T_u, T_u_sum_dist, T_u_color_dist

        def is_fair(T_u_color_dist, T_u_size):
            for color in range(min_nr_color, max_nr_color + 1):
                if T_u_color_dist[color] > (1 + epsilon) * alphas[color] * T_u_size:
                    return False
            return True

        # Order the elements randomly. Note: uncovered is an ordered set.
        node_list = list(uncovered)
        if do_shuffle:
            random.shuffle(node_list)

        for u in node_list:
            logging.debug("current node u is " + str(u))
            T_u, T_u_sum_dist, T_u_color_dist = get_T_u(u)
            logging.debug("|T_u| " + str(len(T_u)) + " color dist " + str(T_u_color_dist))

            # If T_u is fair and dense
            if is_fair(T_u_color_dist, len(T_u)) and T_u_sum_dist <= sigma * len(T_u):
                made_nondegen_cluster = True
                nondegen_clusters.append(T_u)
                nondegen_color_dists.append(T_u_color_dist)
                uncovered -= set(T_u)
                logging.debug(">>>>>>> now uncovered size is " + str(len(uncovered)))
                break

        if made_nondegen_cluster is False:
            break

    clusters = nondegen_clusters + [[u] for u in uncovered]
    res = {"nr_degenerates": len(uncovered),
           "nr_nondegen_clusters": len(nondegen_clusters),
           "nr_clusters": len(clusters),
           "clusters": clusters,
           "nondegen_clusters": nondegen_clusters,
           "degenerates": list(uncovered),
           "nondegen_color_dists": nondegen_color_dists}

    return res


def fair_corr_clustering_driver(graph, lp_res, alphas, epsilon, config):
    # Do grid search on rho and sigma possibly
    nr_rho_step = 5
    nr_sigma_step = 10

    best_res = {}
    best_params = {}
    best_eval_res = {}
    best_eval_res["error_ratio"] = 2  # Note: error_ratio <= 1

    nr_rand_init = config["main"].getint("nr_rand_init")

    # Pre-determined values of rho for grid-search
    rhos = [1 / (2 * nr_rho_step) * i for i in range(1, nr_rho_step + 1)]
    # Will be replaced by config values if given
    if "rhos" in config["main"].keys():
        rhos = [float(i) for i in config["main"].getlist("rhos")]

    for rho in rhos:
        # Pre-determined values of sigma for grid-search
        sigmas = [(rho * i) / (2 * nr_sigma_step) for i in range(1, nr_sigma_step + 1)]
        # Will be replaced by config values if given
        if "sigmas" in config["main"].keys():
            sigmas = [float(i) for i in config["main"].getlist("sigmas")]

        for sigma in sigmas:
            logging.debug("====== Running for rho = " + str(rho) + " and sigma = " + str(sigma))

            # IMPORTANT: Reset the random number generator: We would want the random bits to be equal across
            # different rhos, sigmas, or even different runs of the driver
            random.seed(config["main"].getint("rand_seed"))

            # Do one round of fair_corr_clustering to get some result.
            curr_param_best_res = fair_corr_clustering(graph, lp_res, epsilon, sigma, rho, alphas,
                                                       do_shuffle=(nr_rand_init > 0))
            # Evaluate the result
            curr_param_best_eval_res = eval_clustering(graph, curr_param_best_res, alphas)

            # If supposed to shuffle, repeat and keep the best (w.r.t. the clustering cost)
            for i in range(nr_rand_init - 1):
                new_clustering_res = fair_corr_clustering(graph, lp_res, epsilon, sigma, rho, alphas,
                                                          do_shuffle=True)
                new_eval_res = eval_clustering(graph, new_clustering_res, alphas)

                if new_eval_res["error_ratio"] < curr_param_best_eval_res["error_ratio"]:
                    curr_param_best_res = new_clustering_res
                    curr_param_best_eval_res = new_eval_res
            logging.debug(" <<< best with current setting is: ")
            logging.debug(curr_param_best_eval_res)

            if curr_param_best_eval_res["error_ratio"] < best_eval_res["error_ratio"]:
                best_res = curr_param_best_res
                best_eval_res = curr_param_best_eval_res
                best_params = {"rho": rho, "sigma": sigma}

    return best_eval_res, best_res, best_params
