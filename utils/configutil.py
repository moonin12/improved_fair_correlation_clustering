'''
From https://github.com/nicolasjulioflores/fair_algorithms_for_clustering
Functions that help with reading from a config file.
'''


# Reads the given config string in as a list
#   Allows for multi-line lists.
def read_list(config_string, delimiter=','):
    config_list = config_string.replace("\n", "").split(delimiter)
    return [s.strip() for s in config_list]
