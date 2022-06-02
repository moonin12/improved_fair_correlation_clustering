import json
import logging
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.clustering_utils import get_alphas


def read_csv_data(dataset_name, input_dir, has_index=True, has_header=True, separator=','):
    csv_file = input_dir + dataset_name + ".csv"
    index_col = False
    header = None
    if has_header:
        header = 0
    if has_index:
        index_col = 0
    return pd.read_csv(csv_file, index_col=index_col, header=header, sep=separator)


def write_csv_data(dataset_name, output_dir, df):
    csv_file = output_dir + dataset_name + ".csv"
    df.to_csv(csv_file, index=False)


# Helper function, since json cannot serialize 2d-arrays
def int_list_to_str(list):
    sln = str(int(list[0]))
    for i in range(1, len(list)):
        sln += "," + str(int(list[i]))
    return sln


# dataframe: A pandas dataframe. The first column is the indices, the second is the color, and the rest are the features.
# theta: Top theta fraction of edges with respect to the dot product of the feature vectors are labeled positive
# max_nr_color: The maximum color number that is allowed in the vertices.
# min_nr_color: The minimum color number that is allowed in the vertices.
# def create_graph_form_vectors(df, theta, max_nr_color, min_nr_color, color_columns, vector_columns):
#     all_columns = list(df.keys())
#     logging.debug("column keys: " + str(all_columns))
#     logging.debug("color columns: " + str(color_columns))
#     logging.debug("vector columns: " + str(vector_columns))
#
#     logging.debug("---- dataframe before filtering by color")
#     logging.debug(str(df))
#     # Keep rows with color in [min_nr_color, max_nr_color]
#     for color_col in color_columns:
#         df = df[df[color_col] >= min_nr_color]
#         df = df[df[color_col] <= max_nr_color]
#     # Reset the row indices
#     df.reset_index(drop=True, inplace=True)
#     logging.debug("---- dataframe after filtering by color")
#     logging.debug(str(df))
#
#     nr_nodes = df.shape[0]
#     logging.debug("nr nodes = " + str(nr_nodes))
#     colors = []  # Array of node colors. color[v] is node v's color for all v in range(nr_nodes)
#     nodes_by_color = {}  # Dictionary to split nodes by color. nodes_by_color[i] is a list of all the nodes of color i
#     for color in range(min_nr_color, max_nr_color + 1):
#         nodes_by_color[color] = []
#
#     weighted_edges = []  # List of weighted edges in the format [[w_1,u_1,v_1],...], u_1 < v_1
#     for j, row_2 in df.iterrows():
#         # List of colors this node has
#         color_list = []
#         # For any color feature
#         for color_col in color_columns:
#             color = row_2[color_col]
#             color_list.append(color)
#             nodes_by_color[color].append(j)
#         # Save this list of colors for this node
#         color_str = int_list_to_str(color_list)
#         colors.append(color_str)
#
#         vec_2 = list(row_2[vector_columns])  # Separate the features out of the row; from index 2 onwards
#         for i, row_1 in df.iterrows():  # Loop over vertices i for i < j
#             if i == j:
#                 break
#
#             vec_1 = list(row_1[vector_columns])
#             dot_prod = np.dot(vec_1, vec_2)  # Compute the dot product of feature vectors for nodes i and j
#             logging.debug("dot prod of " + str(i) + " and " + str(j) + " is " + str(dot_prod))
#             weighted_edges.append([dot_prod, i, j])
#
#     logging.debug("---- node colors " + str(colors))
#     logging.debug("---- nodes by color " + str(nodes_by_color))
#     # Sort edges in decreasing weight
#     weighted_edges.sort(reverse=True)
#     logging.debug("---- list of edges with weights, sorted by decreasing weight")
#     logging.debug(str(weighted_edges))
#     # We keep the top theta fraction of the edges as positive edges, hence the definition of nr_positive_edges
#     nr_positive_edges = int(theta * len(weighted_edges))
#     logging.debug("nr positive edges " + str(nr_positive_edges))
#     # positive_edges is the unweighted version of top nr_positive_edges
#     positive_edges = [weighted_edges[i][1:] for i in range(nr_positive_edges)]
#     logging.debug("---- positive edges")
#     logging.debug(str(positive_edges))
#
#     color_dist = {}  # The distribution of colors in the dataset. color_dist[i] = number of nodes of color i in data
#     for color in nodes_by_color.keys():
#         color_dist[color] = len(nodes_by_color[color])
#     logging.debug("---- color distribution " + str(color_dist))
#
#     graph = {"nr_nodes": nr_nodes, "theta": theta, "max_nr_color": max_nr_color, "min_nr_color": min_nr_color,
#              "nr_positive_edges": nr_positive_edges, "color_dist": color_dist, "colors": colors,
#              "nodes_by_color": nodes_by_color, "positive_edges": positive_edges}
#     return graph


# raw_graph: json file of the raw graph
# max_nr_color: The maximum color number that is allowed in the vertices.
# min_nr_color: The minimum color number that is allowed in the vertices.
# sample_per_color: Number of samples we collect from each color in graph
def create_graph_form_vectors(df, theta, max_nr_color, min_nr_color, color_columns, vector_columns, sample_per_color):
    all_columns = list(df.keys())
    logging.debug("column keys: " + str(all_columns))
    logging.debug("color columns: " + str(color_columns))
    logging.debug("vector columns: " + str(vector_columns))

    logging.debug("---- dataframe before filtering by color")
    logging.debug(str(df))
    # Keep rows with color in [min_nr_color, max_nr_color]
    for color_col in color_columns:
        df = df[df[color_col] >= min_nr_color]
        df = df[df[color_col] <= max_nr_color]
    # Reset the row indices
    df.reset_index(drop=True, inplace=True)
    logging.debug("---- dataframe after filtering by color")
    logging.debug(str(df))

    raw_nr_nodes = df.shape[0]
    logging.debug("raw nr nodes = " + str(raw_nr_nodes))

    raw_nodes_by_color = {}
    nodes_by_color = {}  # Dictionary to split nodes by color. nodes_by_color[i] is a list of all the nodes of color i
    for color in range(min_nr_color, max_nr_color + 1):
        raw_nodes_by_color[color] = []
        nodes_by_color[color] = []

    for j, row_2 in df.iterrows():
        for color_col in color_columns:
            color = row_2[color_col]
            raw_nodes_by_color[color].append(j)

    nr_nodes = raw_nr_nodes
    raw_id_map = [i for i in range(raw_nr_nodes)]
    if sample_per_color is not None:
        nr_nodes, raw_id_map = _startified_sampling(raw_nr_nodes, raw_nodes_by_color, min_nr_color, max_nr_color,
                                                    sample_per_color)

    colors = []  # Array of node colors. color[v] is node v's color for all v in range(nr_nodes)

    weighted_edges = []  # List of weighted edges in the format [[w_1,u_1,v_1],...], u_1 < v_1
    for j, row_2 in df.iterrows():
        id_2 = raw_id_map[j]  # new id after down-sampling
        if id_2 < 0:  # Did not sample this row
            continue
        # List of colors this node has
        color_list = []
        # For any color feature
        for color_col in color_columns:
            color = row_2[color_col]
            color_list.append(color)
            nodes_by_color[color].append(id_2)
        # Save this list of colors for this node
        color_str = int_list_to_str(color_list)
        colors.append(color_str)

        vec_2 = list(row_2[vector_columns])  # Separate the features out of the row; from index 2 onwards
        for i, row_1 in df.iterrows():  # Loop over vertices i for i < j
            if i == j:
                break

            id_1 = raw_id_map[i]
            if id_1 < 0:  # Did not sample this row
                continue

            vec_1 = list(row_1[vector_columns])
            dot_prod = np.dot(vec_1, vec_2)  # Compute the dot product of feature vectors for nodes i and j
            logging.debug("dot prod of " + str(id_1) + " and " + str(id_2) + " is " + str(dot_prod))
            weighted_edges.append([dot_prod, id_1, id_2])

    logging.debug("---- node colors " + str(colors))
    logging.debug("---- nodes by color " + str(nodes_by_color))
    # Sort edges in decreasing weight
    weighted_edges.sort(reverse=True)
    logging.debug("---- list of edges with weights, sorted by decreasing weight")
    logging.debug(str(weighted_edges))
    # We keep the top theta fraction of the edges as positive edges, hence the definition of nr_positive_edges
    nr_positive_edges = int(theta * len(weighted_edges))
    logging.debug("nr positive edges " + str(nr_positive_edges))
    # positive_edges is the unweighted version of top nr_positive_edges
    positive_edges = [weighted_edges[i][1:] for i in range(nr_positive_edges)]
    logging.debug("---- positive edges")
    logging.debug(str(positive_edges))

    color_dist = {}  # The distribution of colors in the dataset. color_dist[i] = number of nodes of color i in data
    for color in nodes_by_color.keys():
        color_dist[color] = len(nodes_by_color[color])
    logging.debug("---- color distribution " + str(color_dist))

    graph = {"nr_nodes": nr_nodes, "theta": theta, "max_nr_color": max_nr_color, "min_nr_color": min_nr_color,
             "nr_positive_edges": nr_positive_edges, "color_dist": color_dist, "colors": colors,
             "nodes_by_color": nodes_by_color, "positive_edges": positive_edges}
    return graph


def _startified_sampling(nr_raw_nodes, raw_nodes_by_color, min_nr_color, max_nr_color, sample_per_color):
    for color in range(min_nr_color, max_nr_color + 1):
        logging.debug("loaded " + str(len(raw_nodes_by_color[color])) + " many of color " + str(color))

    # List of raw node samples that will be created now
    sampled_ids = []
    for color in range(min_nr_color, max_nr_color + 1):
        raw_list_of_color = raw_nodes_by_color[color]  # Indices in raw_nodes that have this color
        nr_elts = len(raw_list_of_color)
        # Sample <sample_per_color> many indices of raw_list_of_color
        sampled_inds = random.sample([i for i in range(0, nr_elts)], sample_per_color)
        # Indices in raw_nodes corresponding to this sample
        sampled_ids += [raw_list_of_color[val] for val in sampled_inds]

    sampled_ids.sort()
    nr_nodes = len(sampled_ids)
    logging.debug("sampled " + str(nr_nodes) + " many nodes: " + str(sampled_ids))

    # Map old id's to new id's
    raw_id_map = [-1] * nr_raw_nodes  # -1 indicates this node was not sampled
    for i, val in enumerate(sampled_ids):
        raw_id_map[val] = i
    return nr_nodes, raw_id_map


def create_graph_from_json(raw_graph, max_nr_color, min_nr_color, sample_per_color):
    raw_nodes = raw_graph["nodes"]
    nr_raw_nodes = len(raw_nodes)
    raw_colors = [0] * nr_raw_nodes

    # Create a list of nodes keyed by color
    raw_nodes_by_color = {}  # Dictionary to split nodes by color. nodes_by_color[i] is a list of all the nodes of color i
    nodes_by_color = {}
    for color in range(min_nr_color, max_nr_color + 1):
        raw_nodes_by_color[color] = []
        nodes_by_color[color] = []

    for node in raw_nodes:
        node_id = int(node["id"])
        node_color = int(node["color"])
        raw_colors[node_id] = node_color
        raw_nodes_by_color[node_color].append(node_id)

    nr_nodes = nr_raw_nodes
    raw_id_map = [i for i in range(nr_raw_nodes)]
    if sample_per_color is not None:
        nr_nodes, raw_id_map = _startified_sampling(nr_raw_nodes, raw_nodes_by_color, min_nr_color, max_nr_color,
                                                    sample_per_color)

    # Create colors and nodes_by_color
    colors = [""] * nr_nodes
    for color in range(min_nr_color, max_nr_color + 1):
        for raw_id in raw_nodes_by_color[color]:
            id = raw_id_map[raw_id]  # new id after down-sampling
            if id >= 0:  # Only if inside the sample
                nodes_by_color[color].append(id)
                colors[id] = str(color)

    # Create color_dist
    color_dist = {}
    for color in nodes_by_color.keys():
        color_dist[color] = len(nodes_by_color[color])

    # Only keep positive edges
    positive_edges = []
    for edge in raw_graph["links"]:
        if edge["weight"] == 1:  # Keep edge if positive
            u = raw_id_map[edge["source"]]
            v = raw_id_map[edge["target"]]
            if u >= 0 and v >= 0:  # And both its endpoints are sampled
                positive_edges.append([min(u, v), max(u, v)])  # WARNING: remember the edges are sorted as (u,v), u < v

    nr_positive_edges = len(positive_edges)
    logging.debug("found " + str(nr_positive_edges) + " many positive edges: " + str(positive_edges))

    graph = {"nr_nodes": nr_nodes, "max_nr_color": max_nr_color, "min_nr_color": min_nr_color,
             "nr_positive_edges": nr_positive_edges, "color_dist": color_dist, "colors": colors,
             "nodes_by_color": nodes_by_color, "positive_edges": positive_edges}
    return graph


# WARNING: all dictionary keys are saved as string by json
def write_json(file_name, output_path, content):
    json_file = output_path + file_name + ".json"
    with open(json_file, "w") as jfile:
        json.dump(content, jfile)


# WARNING: all dictionary keys were saved as string by json
def read_json(file_name, input_path):
    json_file = input_path + file_name + ".json"
    opened_file = open(json_file, )
    output = json.loads(opened_file.read())
    opened_file.close()
    return output


def get_instance_name(dataset, theta, min_nr_color, max_nr_color, samp_no=None):
    name = dataset + "_theta" + str(theta) + "_colors" + str(min_nr_color) + "-" + str(max_nr_color)
    if samp_no is not None:
        name += "_samp" + str(samp_no)
    return name


def get_lp_res_name(graph_name, alphas_def, scale_id):
    return graph_name + "_alphas" + str(alphas_def) + "_scale" + str(scale_id)


def get_clustering_name(instance_name, epsilon, sigma, rho):
    return instance_name + "_epsilon" + str(epsilon) + "_sigma" + str(sigma) + "_rho" + str(rho)


# Read json file, but convert the dictionary keys to integers. str_keys are the names of the dictionary fields.
def read_json_int_keys(file_name, input_path, str_keys, str_values_keys=[]):
    contents = read_json(file_name, input_path)

    def convert_dict_key_to_int(target_dict):
        nonlocal contents
        org_dict = contents[target_dict]
        int_dict = {}
        for key, value in org_dict.items():
            int_dict[int(key)] = value
        contents[target_dict] = int_dict

    def convert_string_val_to_int(target_list):
        nonlocal contents
        org_list = contents[target_list]
        int_list = []
        for elt in org_list:
            elt_split = elt.split(',')
            int_split = []
            for token in elt_split:
                int_split.append(int(token))
            int_list.append(int_split)
        contents[target_list] = int_list

    for key in str_keys:
        convert_dict_key_to_int(key)

    for key in str_values_keys:
        convert_string_val_to_int(key)

    return contents


def line_plot(benchmarks, values, stds, keys, plot_y_label, plot_x_label, plot_title, data_dir, plot_name):
    benchmark_styles = {"Fair-CC": ["-go", "red"],
                        "Fair LP": ["-k", "black"],
                        }

    fig, ax = plt.subplots()
    for benchmark in benchmarks:
        value_array = values[benchmark]
        if len(value_array) != len(keys):
            print("WARNING dim doesn't match! " + str(len(value_array)))

        if stds is not None:
            std_array = stds[benchmark]
            ax.errorbar(keys, value_array, fmt=benchmark_styles[benchmark][0], yerr=std_array, label=benchmark, capsize=6,
                        markersize=12)
        else:
            ax.plot(keys, value_array, benchmark_styles[benchmark][0], label=benchmark, markersize=12)

    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=18)
    if len(benchmarks) == 1:
        lgd.remove()
    plt.xticks(keys, keys)

    plt.ylabel(plot_y_label, fontsize=18)
    plt.xlabel(plot_x_label, fontsize=18)
    # plt.title(plot_title, fontsize=18)

    plt.savefig(data_dir + plot_name + ".png", bbox_extra_artists=(lgd,),
                bbox_inches='tight')
    plt.clf()


def read_graph_and_lp(dataset, theta, samp_no, scale_id, config):
    min_nr_color = config[dataset].getint("min_nr_color")
    max_nr_color = config[dataset].getint("max_nr_color")
    graph_name = get_instance_name(dataset, theta, min_nr_color, max_nr_color, samp_no)
    graph_dir = config["main"].get("graph_dir")
    alphas_def = config["main"].getint("alphas_def")

    graph = read_json_int_keys(graph_name, graph_dir, ["color_dist", "nodes_by_color"], ["colors"])
    alphas = get_alphas(graph["color_dist"], alphas_def)
    logging.debug("----- graph_name " + graph_name + " alphas are " + str(alphas))

    lp_res_dir = config["main"].get("lp_res_dir")
    lp_res_name = get_lp_res_name(graph_name, alphas_def, scale_id)
    lp_res = read_json(lp_res_name, lp_res_dir)

    return graph, lp_res, lp_res_name, alphas, graph_name
