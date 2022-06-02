import configparser
import logging

from fair_corr_lp_solver import solve_fair_corr_lp
from utils.clustering_utils import get_alphas, get_scaled_alphas
from utils.configutil import read_list
from utils.read_write_utils import get_instance_name, get_lp_res_name, read_json_int_keys, write_json

logger = logging.getLogger()


# logger.setLevel(logging.DEBUG)

# config_file = "configs/config-prevwork-compare-vectors-2colors.ini"
# Read config file

def lp_solving_script(config_file):
    config = configparser.ConfigParser(converters={'list': read_list})
    config.read(config_file)

    graph_dir = config["main"].get("graph_dir")

    vector_datasets = []
    thetas = []
    if "vector_datasets" in config["main"].keys():
        vector_datasets = config["main"].getlist("vector_datasets")
        thetas = [float(i) for i in config["main"].getlist("thetas")]

    graph_datasets = []
    if "graph_datasets" in config["main"].keys():
        graph_datasets = config["main"].getlist("graph_datasets")

    datasets = vector_datasets + graph_datasets

    alphas_def = config["main"].getint("alphas_def")
    lp_solving_method = config["main"].getint("lp_solving_method")
    lp_res_dir = config["main"].get("lp_res_dir")

    samp_nos = [None]
    if "nr_sub_samp" in config["main"].keys():
        nr_sub_samp = config["main"].getint("nr_sub_samp")
        samp_nos = [i for i in range(nr_sub_samp)]

    alpha_scale_step = config["main"].getint("alpha_scale_step")

    for dataset in datasets:
        current_thetas = thetas
        if dataset in graph_datasets:
            current_thetas = [-1]  # convention to set theta to -1 for graph datasets
        for theta in current_thetas:
            min_nr_color = config[dataset].getint("min_nr_color")
            max_nr_color = config[dataset].getint("max_nr_color")

            for samp_no in samp_nos:
                graph_name = get_instance_name(dataset, theta, min_nr_color, max_nr_color, samp_no)

                graph = read_json_int_keys(graph_name, graph_dir, ["color_dist", "nodes_by_color"], ["colors"])
                alphas = get_alphas(graph["color_dist"], alphas_def)

                all_scaled_alphas, scale_factors = get_scaled_alphas(alphas, alpha_scale_step)
                for i in range(alpha_scale_step):
                    scaled_alphas = all_scaled_alphas[i]
                    lp_res = solve_fair_corr_lp(graph, scaled_alphas, lp_solving_method)

                    lp_res_name = get_lp_res_name(graph_name, alphas_def, i)
                    write_json(lp_res_name, lp_res_dir, lp_res)
                    logger.debug("------ for lp res name " + str(lp_res_name))
                    logger.debug("scaled alphas are: ")
                    logger.debug(scaled_alphas)


# # Solve LP's for:
# # Varying alphas, epsilon is irrelevant (it'll get fixed to 0.01), two colors, ONE sumbsample on amazon, not sumbsampled, reuters and victorian
# lp_solving_script("configs/config-prevwork-compare-all-nosamp-2colors.ini")
# # Fixed alphas, epsilon is irrelevant, two colors, 10 subsamples on amazon
# lp_solving_script("configs/config-prevwork-compare-amazon-2colors.ini")
# # Fixed alphas, epsilon is irrelevant, two colors, not sumbsampled, reuters and victorian
# lp_solving_script("configs/config-prevwork-compare-vectors-2colors.ini")
# # Fixed alphas, epsilon is irrelevant, four colors, sumbsampled, reuters and victorian
# lp_solving_script("configs/config-prevwork-compare-4colors.ini")
# # Fixed alphas, epsilon is irrelevant, eight colors, sumbsampled, reuters and victorian
# lp_solving_script("configs/config-prevwork-compare-8colors.ini")
# # Fixed alphas, epsilon is irrelevant, all colors, bank_200, census_200, and diabetes_200 (each has 200 (+1) nodes)
# lp_solving_script("configs/config-overlapping-colors.ini")
