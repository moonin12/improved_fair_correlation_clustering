This project includes codes and data of experiments for "Improved Approximation for Fair Correlation Clustering" by Sara Ahmadian and Maryam Negahbani
See the manuscript for algorithm and data descriptions.

The main fair clustering scripts are in fair_clustering_script.py. See the end of the file for instructions on which scripts to run for which experiment.
Each experiment reads the corresponding graph and LP solution from ./graphs and ./lp_res_color_dist respectively.
The ./graphs directory is populated with the graphs and subsamples used in our experiments. See graph_generation_script.py for a description of how they are generated.
The LP solutions in ./lp_res_color_dist are generated using lp_solving_script.py. See the end of the file for instructions on which script to run for which experiment. CPLEX libraries are required.
All the mentioned scripts rely on config files in ./configs that set the parameters for different experimental setups.
