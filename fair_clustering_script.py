import configparser
import logging
import random

import numpy as np

from fair_corr_clustering import fair_corr_clustering_driver
from utils.clustering_utils import get_scaled_alphas
from utils.configutil import read_list
from utils.read_write_utils import read_graph_and_lp, line_plot, write_json, read_json, get_instance_name, \
    get_lp_res_name

logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)

config_file = "configs/config-prevwork-compare-amazon-2colors.ini"
# config_file = "configs/config-prevwork-compare-vectors-2colors.ini"
# config_file = "configs/config-prevwork-compare-all-nosamp-2colors.ini"

# Read config file
config = configparser.ConfigParser(converters={'list': read_list})


def run_prevwork_comparison_experiments(config_file):
    config.read(config_file)

    vector_datasets = []
    thetas = []
    if "vector_datasets" in config["main"].keys():
        vector_datasets = config["main"].getlist("vector_datasets")
        thetas = [float(i) for i in config["main"].getlist("thetas")]

    graph_datasets = []
    if "graph_datasets" in config["main"].keys():
        graph_datasets = config["main"].getlist("graph_datasets")

    datasets = vector_datasets + graph_datasets

    random.seed(config["main"].getint("rand_seed"))

    # WARNING: We run only on fixed epsilon, set to 0.01 if not provided. We do NOT run on scaled alphas
    epsilon = 0.01
    samp_nos = [None]
    if "nr_sub_samp" in config["main"].keys():
        nr_sub_samp = config["main"].getint("nr_sub_samp")
        samp_nos = [i for i in range(nr_sub_samp)]

    for dataset in datasets:
        current_thetas = thetas
        if dataset in graph_datasets:
            current_thetas = [-1]  # convention set theta to -1 for graph datasets
        for theta in current_thetas:
            costs = []
            lp_costs = []
            max_violations = []
            nr_degenerates = []
            rhos = []
            sigmas = []
            for samp_no in samp_nos:
                graph, lp_res, lp_res_name, alphas, graph_name = read_graph_and_lp(dataset, theta, samp_no=samp_no,
                                                                                   scale_id=0, config=config)
                nr_nodes = graph["nr_nodes"]
                nr_edges = (nr_nodes * (nr_nodes - 1)) / 2
                lp_cost_ratio = lp_res["cost"] / nr_edges
                best_eval_res, best_clustering_res, best_params = fair_corr_clustering_driver(graph, lp_res, alphas,
                                                                                              epsilon, config)
                costs.append(best_eval_res["error_ratio"])
                lp_costs.append(lp_cost_ratio)
                max_violations.append(best_eval_res["max_violation"])
                nr_degenerates.append(best_eval_res["nr_degenerates"])
                rhos.append(best_params["rho"])
                sigmas.append(best_params["sigma"])

            print("------------------------ results for experiment " + graph_name)
            print(costs)
            print("Average cost: " + str(np.average(costs)))
            print("STD of cost: " + str(np.std(costs)))
            print(lp_costs)
            print(max_violations)
            print(nr_degenerates)
            print(rhos)
            print(sigmas)


def run_varying_epsilon_experiments(confg_file):
    config.read(config_file)

    vector_datasets = []
    thetas = []
    if "vector_datasets" in config["main"].keys():
        vector_datasets = config["main"].getlist("vector_datasets")
        thetas = [float(i) for i in config["main"].getlist("thetas")]

    graph_datasets = []
    if "graph_datasets" in config["main"].keys():
        graph_datasets = config["main"].getlist("graph_datasets")

    datasets = vector_datasets + graph_datasets
    plot_dir = config["main"].get("plot_dir")

    random.seed(config["main"].getint("rand_seed"))

    # WARNING: we do NOT run these experiments for varying alphas
    epsilons = [float(i) for i in config["main"].getlist("epsilons")]

    samp_nos = [None]
    if "nr_sub_samp" in config["main"].keys():
        nr_sub_samp = config["main"].getint("nr_sub_samp")
        samp_nos = [i for i in range(nr_sub_samp)]

    for dataset in datasets:
        current_thetas = thetas
        if dataset in graph_datasets:
            current_thetas = [-1]  # convention set theta to -1 for graph datasets
        for theta in current_thetas:
            for samp_no in samp_nos:
                rhos = []
                sigmas = []

                error_ratios = []
                max_violations = []
                nr_degenerates = []
                lp_cost_ratios = []

                graph, lp_res, lp_res_name, alphas, graph_name = read_graph_and_lp(dataset, theta, samp_no,
                                                                                   scale_id=0, config=config)
                nr_nodes = graph["nr_nodes"]
                nr_edges = (nr_nodes * (nr_nodes - 1)) / 2
                lp_cost_ratio = lp_res["cost"] / nr_edges

                for epsilon in epsilons:
                    best_eval_res, best_clustering_res, best_params = fair_corr_clustering_driver(graph, lp_res, alphas,
                                                                                                  epsilon, config)
                    print("------------------------ results for experiment " + graph_name)
                    print(best_params)
                    rhos.append(best_params["rho"])
                    sigmas.append(best_params["sigma"])

                    print(best_eval_res)
                    error_ratios.append(best_eval_res["error_ratio"])
                    max_violations.append(best_eval_res["max_violation"])
                    nr_degenerates.append(best_eval_res["nr_degenerates"])
                    lp_cost_ratios.append(lp_cost_ratio)

                output = {"rhos": rhos, "sigmas": sigmas, "error_ratios": error_ratios,
                          "max_violations": max_violations,
                          "nr_degenerates": nr_degenerates, "lp_cost_ratios": lp_cost_ratios}
                write_json(lp_res_name + "_var_eps", plot_dir, output)


def plot_varying_epsilon_experiments(config_file):
    config.read(config_file)

    vector_datasets = []
    thetas = []
    if "vector_datasets" in config["main"].keys():
        vector_datasets = config["main"].getlist("vector_datasets")
        thetas = [float(i) for i in config["main"].getlist("thetas")]

    graph_datasets = []
    if "graph_datasets" in config["main"].keys():
        graph_datasets = config["main"].getlist("graph_datasets")

    datasets = vector_datasets + graph_datasets
    plot_dir = config["main"].get("plot_dir")

    keys = [float(i) for i in config["main"].getlist("epsilons")]

    for dataset in datasets:
        current_thetas = thetas
        if dataset in graph_datasets:
            current_thetas = [-1]  # convention set theta to -1 for graph datasets
        for theta in current_thetas:
            costs = {"Fair-CC": {}, "Fair LP": {}}
            fairness = {"Fair-CC": {}}

            for epsilon in keys:
                costs["Fair-CC"][epsilon] = []
                costs["Fair LP"][epsilon] = []
                fairness["Fair-CC"][epsilon] = []

            samp_nos = [None]
            if "nr_sub_samp" in config["main"].keys():
                nr_sub_samp = config["main"].getint("nr_sub_samp")
                samp_nos = [i for i in range(nr_sub_samp)]

            for samp_no in samp_nos:
                min_nr_color = config[dataset].getint("min_nr_color")
                max_nr_color = config[dataset].getint("max_nr_color")
                graph_name = get_instance_name(dataset, theta, min_nr_color, max_nr_color, samp_no)
                lp_res_name = get_lp_res_name(graph_name, config["main"].getint("alphas_def"), scale_id=0)
                res = read_json(lp_res_name + "_var_eps", plot_dir)

                for i, epsilon in enumerate(keys):
                    costs["Fair-CC"][epsilon].append(res["error_ratios"][i])
                    costs["Fair LP"][epsilon].append(res["lp_cost_ratios"][i])
                    fairness["Fair-CC"][epsilon].append(res["max_violations"][i])

            costs_avg = {"Fair-CC": [], "Fair LP": []}
            fairness_avg = {"Fair-CC": []}

            costs_std = {"Fair-CC": [], "Fair LP": []}
            fairness_std = {"Fair-CC": []}
            for epsilon in keys:
                costs_avg["Fair-CC"].append(np.average(costs["Fair-CC"][epsilon]))
                costs_avg["Fair LP"].append(np.average(costs["Fair LP"][epsilon]))
                fairness_avg["Fair-CC"].append(np.average(fairness["Fair-CC"][epsilon]))

                costs_std["Fair-CC"].append(np.std(costs["Fair-CC"][epsilon]))
                costs_std["Fair LP"].append(np.std(costs["Fair LP"][epsilon]))
                fairness_std["Fair-CC"].append(np.std(fairness["Fair-CC"][epsilon]))

            line_plot(["Fair LP", "Fair-CC"], costs_avg, costs_std, keys, "cost ratio", "ε",
                      "Cost comparison with opt, varying ε",
                      plot_dir, lp_res_name + "_lp_var_eps")
            line_plot(["Fair-CC"], fairness_avg, fairness_std, keys, "maximum violation", "ε",
                      "Maximum fairness violation, varying ε",
                      plot_dir, lp_res_name + "_viol_var_eps")


def run_varying_min_alpha_experiments(config_file):
    config.read(config_file)

    vector_datasets = []
    thetas = []
    if "vector_datasets" in config["main"].keys():
        vector_datasets = config["main"].getlist("vector_datasets")
        thetas = [float(i) for i in config["main"].getlist("thetas")]

    graph_datasets = []
    if "graph_datasets" in config["main"].keys():
        graph_datasets = config["main"].getlist("graph_datasets")

    datasets = vector_datasets + graph_datasets
    plot_dir = config["main"].get("plot_dir")

    random.seed(config["main"].getint("rand_seed"))

    # WARNING: We fix epsilon = 0.01 instead of reading from config.
    epsilon = 0.01
    # WARNING: We do not run on multiple subsamples
    samp_no = None
    alpha_scale_step = config["main"].getint("alpha_scale_step")

    for dataset in datasets:
        current_thetas = thetas
        if dataset in graph_datasets:
            current_thetas = [-1]  # convention to set theta to -1 for graph datasets
        for theta in current_thetas:
            rhos = []
            sigmas = []

            error_ratios = []
            max_violations = []
            nr_degenerates = []
            lp_cost_ratios = []
            scaled_min_alphas = []

            for i in range(alpha_scale_step):
                graph, lp_res, lp_res_name, alphas, graph_name = read_graph_and_lp(dataset, theta, samp_no,
                                                                                   scale_id=i, config=config)

                nr_nodes = graph["nr_nodes"]
                nr_edges = (nr_nodes * (nr_nodes - 1)) / 2
                lp_cost_ratio = lp_res["cost"] / nr_edges

                all_scaled_alphas, scale_factors = get_scaled_alphas(alphas, alpha_scale_step)
                scaled_alphas = all_scaled_alphas[i]

                scaled_min_alphas.append(float("{:.2f}".format(min(scaled_alphas.values()))))
                best_eval_res, best_clustering_res, best_params = fair_corr_clustering_driver(graph, lp_res,
                                                                                              scaled_alphas,
                                                                                              epsilon, config)
                print("------------------------ results for experiment " + graph_name)
                print(" i is " + str(i) + " and scaled alphas are: ")
                print(scaled_alphas)

                print(best_params)
                rhos.append(best_params["rho"])
                sigmas.append(best_params["sigma"])

                print(best_eval_res)
                error_ratios.append(best_eval_res["error_ratio"])
                max_violations.append(best_eval_res["max_violation"])
                nr_degenerates.append(best_eval_res["nr_degenerates"])
                lp_cost_ratios.append(lp_cost_ratio)

            output = {"rhos": rhos, "sigmas": sigmas, "error_ratios": error_ratios,
                      "max_violations": max_violations,
                      "nr_degenerates": nr_degenerates, "lp_cost_ratios": lp_cost_ratios,
                      "scaled_min_alphas": scaled_min_alphas}
            write_json(lp_res_name + "_var_minalphas", plot_dir, output)


def plot_varying_min_alpha_experiments(config_file):
    config.read(config_file)

    vector_datasets = []
    thetas = []
    if "vector_datasets" in config["main"].keys():
        vector_datasets = config["main"].getlist("vector_datasets")
        thetas = [float(i) for i in config["main"].getlist("thetas")]

    graph_datasets = []
    if "graph_datasets" in config["main"].keys():
        graph_datasets = config["main"].getlist("graph_datasets")

    datasets = vector_datasets + graph_datasets
    plot_dir = config["main"].get("plot_dir")

    # WARNING: We fix epsilon = 0.01 instead of reading from config.
    epsilon = 0.01
    # WARNING: We do not run on multiple subsamples
    samp_no = None
    alpha_scale_step = config["main"].getint("alpha_scale_step")

    for dataset in datasets:
        current_thetas = thetas
        if dataset in graph_datasets:
            current_thetas = [-1]  # convention to set theta to -1 for graph datasets
        for theta in current_thetas:
            i = alpha_scale_step - 1
            min_nr_color = config[dataset].getint("min_nr_color")
            max_nr_color = config[dataset].getint("max_nr_color")
            graph_name = get_instance_name(dataset, theta, min_nr_color, max_nr_color, samp_no)
            lp_res_name = get_lp_res_name(graph_name, config["main"].getint("alphas_def"), scale_id=i)
            res = read_json(lp_res_name + "_var_minalphas", plot_dir)

            keys = res["scaled_min_alphas"]
            values = {"Fair-CC": res["error_ratios"], "Fair LP": res["lp_cost_ratios"]}
            plot_name = lp_res_name + "_var_minalphas"
            line_plot(["Fair LP", "Fair-CC"], values, None, keys, "cost ratio", "$α^{min}$",
                      "Cost comparison with opt, varying $α^{min}$",
                      plot_dir, plot_name)

# # -------------------  run_prevwork_comparison_experiments for:
# # Fixed alphas, epsilon is irrelevant (it'll get fixed to 0.01), two colors, 10 subsamples on amazon
# run_prevwork_comparison_experiments("configs/config-prevwork-compare-amazon-2colors.ini")
# # Fixed alphas, epsilon is irrelevant (it'll get fixed to 0.01), two colors, not sumbsampled, reuters and victorian
# run_prevwork_comparison_experiments("configs/config-prevwork-compare-vectors-2colors.ini")
# # Fixed alphas, epsilon is irrelevant (it'll get fixed to 0.01), four colors, sumbsampled, reuters and victorian
# run_prevwork_comparison_experiments("configs/config-prevwork-compare-4colors.ini")
# # Fixed alphas, epsilon is irrelevant (it'll get fixed to 0.01), eight colors, sumbsampled, reuters and victorian
# run_prevwork_comparison_experiments("configs/config-prevwork-compare-8colors.ini")
# # Fixed alphas, epsilon is irrelevant (it'll get fixed to 0.01), all colors, bank_200, census_200, and diabetes_200 (each has 200 (+1) nodes)
# run_prevwork_comparison_experiments("configs/config-overlapping-colors.ini")

# # -------------------  run_varying_epsilon_experiments for:
# # Fixed alphas, varying epsilon, two colors, 10 subsamples on amazon
# run_varying_epsilon_experiments("configs/config-prevwork-compare-amazon-2colors.ini")
# plot_varying_epsilon_experiments("configs/config-prevwork-compare-amazon-2colors.ini")
# # Fixed alphas, varying epsilon, two colors, not sumbsampled, reuters and victorian
# run_varying_epsilon_experiments("configs/config-prevwork-compare-vectors-2colors.ini")
# plot_varying_epsilon_experiments("configs/config-prevwork-compare-vectors-2colors.ini")

# # -------------------  run_varying_min_alpha_experiments for:
# # Varying alphas, epsilon is irrelevant (it'll get fixed to 0.01), two colors, ONE sumbsample on amazon, not sumbsampled, reuters and victorian
# run_varying_min_alpha_experiments("configs/config-prevwork-compare-all-nosamp-2colors.ini")
# plot_varying_min_alpha_experiments("configs/config-prevwork-compare-all-nosamp-2colors.ini")
