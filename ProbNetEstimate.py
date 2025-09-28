def main():
    import os

    structure_file = "naive_bayes.str"
    data_file = "spam_test.csv"
    output_file = "output.txt"

    if not os.path.isfile(structure_file):
        logging.error(f"Structure file '{structure_file}' not found. Please place it in the folder.")
        return

    if not os.path.isfile(data_file):
        logging.warning(f"Data file '{data_file}' not found. Safe mode will run with pseudo-counts only.")

    logging.info(f"Parsing Bayesian network structure from '{structure_file}'...")
    nodes = BayesianNetwork.parse_structure(structure_file)

    logging.info(f"Learning CPTs from training data '{data_file}' (safe mode)...")
    nodes = BayesianNetwork.parse_data(data_file, nodes)

    logging.info(f"Writing CPTs to '{output_file}'...")
    BayesianNetwork.write_nodes_cpt(nodes, filename=output_file)

    logging.info("CPT learning completed successfully.")