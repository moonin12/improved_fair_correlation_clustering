import logging


# clusters: List of clusters where each cluster is a list of nodes (from 0 to <nr_nodes> - 1)
# nr_nodes: Total number of nodes
# Returns array cluster_inds where for each node v cluster_inds[v] is the index of v's cluster
def get_cluster_inds_from_clusters(clusters, nr_nodes):
    cluster_inds = [0] * nr_nodes
    for cluster_ind in range(len(clusters)):
        cluster = clusters[cluster_ind]
        for node in cluster:
            cluster_inds[node] = cluster_ind
    return cluster_inds


# clusters: List of clusters where each cluster is a list of nodes (from 0 to <nr_nodes> - 1)
# graph: Graph dictionary
# returns the clustering error ratio which is the clustering error divided by total number of edges
def get_corr_clustering_error(clusters, graph):
    nr_nodes = graph["nr_nodes"]
    cluster_inds = get_cluster_inds_from_clusters(clusters, nr_nodes)
    positive_edges = graph["positive_edges"]
    # Number of positive edges inside each cluster
    nr_pos_in_cluster = [0] * len(clusters)
    # Number of positive edges going across
    nr_pos_across = 0

    # Fill in nr_pos_in_cluster and nr_pos_across
    for edge in positive_edges:
        u = edge[0]
        v = edge[1]
        if cluster_inds[u] == cluster_inds[v]:
            nr_pos_in_cluster[cluster_inds[u]] += 1
        else:
            nr_pos_across += 1

    # Total number of negative edges inside clusters
    nr_neg_inside = 0
    for cluster_ind in range(len(clusters)):
        cluster_size = len(clusters[cluster_ind])
        all_edges_inside = (cluster_size * (cluster_size - 1)) / 2
        nr_neg_inside += all_edges_inside - nr_pos_in_cluster[cluster_ind]

    nr_edges = (nr_nodes * (nr_nodes - 1)) / 2
    return (nr_pos_across + nr_neg_inside) / nr_edges


# Get the clustering fairness violation defined as max_violation := {max_C max_i  |V_i \cap C|/(\alpha_i |C|)} - 1
# By definition, max_violation <= epsilon
def get_corr_clustering_fairness_viol(nr_nodes, nondegen_clusters, nondegen_color_dists, alphas):
    # max_violation := {max_C max_i  |V_i \cap C|/(\alpha_i |C|)} - 1
    max_violation = 0
    logging.debug("----- get corr clustering max_violation with nodes " + str(nr_nodes))
    logging.debug("alphas : " + str(alphas))
    for cluster_ind in range(len(nondegen_clusters)):
        logging.debug("*** non_degen cluster ind " + str(cluster_ind))
        cluster_size = len(nondegen_clusters[cluster_ind])
        cluster_dist = nondegen_color_dists[cluster_ind]
        logging.debug("size " + str(cluster_size) + " dist " + str(cluster_dist))
        for color in cluster_dist.keys():
            this_violation = cluster_dist[color] / (cluster_size * alphas[int(color)]) - 1
            logging.debug("violation of color " + str(color) + " is " + str(this_violation))
            max_violation = max(max_violation, this_violation)
    return max_violation


def eval_clustering(graph, clustering_res, alphas):
    clusters = clustering_res["clusters"]
    nondegen_clusters = clustering_res["nondegen_clusters"]
    degenerates = clustering_res["degenerates"]
    error_ratio = get_corr_clustering_error(clusters, graph)
    max_violation = get_corr_clustering_fairness_viol(graph["nr_nodes"], nondegen_clusters,
                                                      clustering_res["nondegen_color_dists"], alphas)

    res = {"error_ratio": error_ratio, "max_violation": max_violation,
           "nr_degenerates": len(clustering_res["degenerates"])}
    return res


# Get alpha values corresponding to definition given by color_caps_def and according to color distribution in color_dist
def get_alphas(color_dist, color_caps_def):
    color_caps = {}

    # color_caps = ratio of nodes from that color in the whole dataset
    if color_caps_def == 1:
        nr_colors = len(color_dist.keys())
        nr_nodes = 0
        for color in color_dist.keys():
            nr_nodes += color_dist[color]
        for color in color_dist.keys():
            color_caps[color] = max(color_dist[color] / nr_nodes, 1 / nr_colors)

    # color_caps is 1/2 for all colors
    if color_caps_def == 2:
        for color in color_dist.keys():
            color_caps[color] = 0.5

    # color_caps is 0.8 for all colors
    if color_caps_def == 3:
        for color in color_dist.keys():
            color_caps[color] = 0.8

    return color_caps


# Gets scaled alphas based on scale factors. Returns all_scaled_alphas which is an array of <alpha_scale_step> many
#  dictionaries. Each dictionary is one set of scaled alphas
def get_scaled_alphas(alphas, alpha_scale_step):
    if alpha_scale_step == 1:
        return [alphas], [1]
    min_alpha = min(alphas.values())
    # logger.debug("----- min alpha " + str(min_alpha) + " alphas are ")
    # logger.debug(alphas)
    # Scale factors range from 1 (do nothing) to 1/min_alpha (removes fairness)
    scale_fact_range = 1 - 1 / min_alpha
    # say alpha_scale_step = 6 (i is from 0 to 5)
    # i = 0: 1/min_alpha -> practically removes fairness constraints
    # i = 1: 1/min_alpha + (1 - 1/min_alpha)/5 when mult by min_alpha -> 1 + min_alpha/5 - 1/5 between 4/5 and 1
    # i = 5: 1/min_alpha + (1 - 1/min_alpha) = 1 -> do nothing
    scale_factors = [1 / min_alpha + (scale_fact_range * i) / (alpha_scale_step - 1) for i in range(0, alpha_scale_step)]
    scale_factors.sort()

    all_scaled_alphas = []
    for scale_fact in scale_factors:
        scaled_alphas = {}
        for color in alphas.keys():
            scaled_alphas[color] = min(1, scale_fact * alphas[color])
        all_scaled_alphas.append(scaled_alphas)
    return all_scaled_alphas, scale_factors
