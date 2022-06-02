import configparser

from utils.configutil import read_list
from utils.read_write_utils import read_csv_data, create_graph_form_vectors, create_graph_from_json, get_instance_name, \
    write_json, read_json


# logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)

def graph_generation_script(config_file):
    # Read config file
    config = configparser.ConfigParser(converters={'list': read_list})
    config.read(config_file)

    data_dir = config["main"].get("data_dir")
    graph_dir = config["main"].get("graph_dir")

    samp_nos = [None]
    if "nr_sub_samp" in config["main"].keys():
        nr_sub_samp = config["main"].getint("nr_sub_samp")
        samp_nos = [i for i in range(nr_sub_samp)]

    if "vector_datasets" in config["main"].keys():
        vector_datasets = config["main"].getlist("vector_datasets")
        thetas = [float(i) for i in config["main"].getlist("thetas")]

        for dataset in vector_datasets:
            has_header = config[dataset].getboolean("has_header")
            has_index = config[dataset].getboolean("has_index")
            color_columns = config[dataset].getlist("color_columns")
            vector_columns = config[dataset].getlist("vector_columns")
            min_nr_color = config[dataset].getint("min_nr_color")
            max_nr_color = config[dataset].getint("max_nr_color")
            sample_for_color = None
            if "sample_for_color" in config[dataset].keys():
                sample_for_color = config[dataset].getint("sample_for_color")
            for theta in thetas:
                df = read_csv_data(dataset, data_dir, has_index=has_index, has_header=has_header)
                if not has_header:
                    df.columns = [str(i) for i in df.keys()]

                for samp_no in samp_nos:
                    graph = create_graph_form_vectors(df, theta, max_nr_color, min_nr_color, color_columns,
                                                      vector_columns, sample_for_color)
                    graph_name = get_instance_name(dataset, theta, min_nr_color, max_nr_color, samp_no)
                    write_json(graph_name, graph_dir, graph)

    if "graph_datasets" in config["main"].keys():
        graph_datasets = config["main"].getlist("graph_datasets")
        for dataset in graph_datasets:
            theta = -1  # Convention
            min_nr_color = config[dataset].getint("min_nr_color")
            max_nr_color = config[dataset].getint("max_nr_color")
            sample_for_color = config[dataset].getint("sample_for_color")

            raw_graph = read_json(dataset, data_dir)
            for samp_no in samp_nos:
                graph = create_graph_from_json(raw_graph, max_nr_color, min_nr_color, sample_for_color)
                graph_name = get_instance_name(dataset, theta, min_nr_color, max_nr_color, samp_no)
                write_json(graph_name, graph_dir, graph)

# # Run graph_generation_script on:
# graph_generation_script("configs/config-prevwork-compare-amazon-2colors.ini")
# graph_generation_script("configs/config-prevwork-compare-vectors-2colors.ini")
# graph_generation_script("configs/config-prevwork-compare-4colors.ini")
# graph_generation_script("configs/config-prevwork-compare-8colors.ini")
# graph_generation_script("configs/config-overlapping-colors.ini")
