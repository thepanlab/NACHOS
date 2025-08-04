from util.get_config import parse_json
from termcolor import colored
from statistics import mode
import numpy as np
import pandas as pd
import regex as re
import math
import os


def get_input_matrices(matrices_path, is_outer):
    """ Finds the existing configs, test folds, and validations folds of all matrices.

    Args:
        matrices_path (string): The path of the matrices' directory.
        is_outer (bool): Whether this data is of the outer loop.
        is_outer (bool): Whether this data is of the outer loop.

    Returns:
        dict: A dictionary of matrix-dataframes amd their prediction shapes, organized by config and testing fold.
    """
    # Get the confusion matrix paths that were created earlier
    try:
        all_paths = os.listdir(matrices_path)
    except:
        print(colored("Error: could not read matrices path. Are you sure it's correct?", 'red'))
        exit(-1)
    organized_paths = {}
    organized_shapes = {}

    # Separate by config
    for path in all_paths:
        config = path.split("/")[0].split('_')[0]
        if config not in organized_paths:
            organized_paths[config] = []
        organized_paths[config].append(path)

    # For each config, separate by testing fold
    for config in organized_paths:
        config_paths = organized_paths[config]
        organized_paths[config] = {}
        organized_shapes[config] = {}
        for path in config_paths:
            filename = path.split('/')[-1]
                
            # Search for the test-fold name from the file name
            if not is_outer:
                try:
                    test_fold = re.search('_test_.*_val_', filename).captures()[0].split("_")[2] 
                except:    
                    raise ValueError(colored(f"Error: No test-val pair found in the filename {filename}\n\tAre you sure it is of the inner loop?",'red'))       
                try:
                    test_fold = re.search('_test_.*_val_', filename).captures()[0].split("_")[2] 
                except:    
                    raise ValueError(colored(f"Error: No test-val pair found in the filename {filename}\n\tAre you sure it is of the inner loop?",'red'))       
                if test_fold not in organized_paths[config]:
                    organized_paths[config][test_fold] = {}
                    organized_shapes[config][test_fold] = {}
            else:
                test_fold = re.search('_test_.*_', filename).captures()[0].split("_")[2]
                test_fold = re.search('_test_.*_', filename).captures()[0].split("_")[2]
                
            # Search for the val-fold name from the file name, read the csv, and get shape
            if not is_outer:
                val_fold = re.search('_test_.*_val_', filename).captures()[0].split("_")[4]
                shape = re.findall(r'\d+', filename)[-1]
                organized_shapes[config][test_fold][val_fold] = shape
                organized_paths[config][test_fold][val_fold] = os.path.join(matrices_path, path)
            else:
                shape = int(re.search('_.*_conf_matrix.csv', filename).captures()[0].split("_")[1])
                organized_shapes[config][test_fold] = shape
                organized_paths[config][test_fold] = os.path.join(matrices_path, path)

    # Return the dictionary of organized matrices
    return organized_paths, organized_shapes


def get_matrices_of_mode_shape(shapes, matrices, is_outer):
    """ Finds the mode length of all predictions that exist within a data folder.
        Checks if matrices should be considered in the mean value.

    Args:
        shapes (dict): The shapes (prediction rows) of the corresponding confusion matrices.
        matrices (dict): A dictionary of the matrices.
        is_outer (bool): Whether this data is of the outer loop.
        is_outer (bool): Whether this data is of the outer loop.

    Returns:
        dict: A reduced dictionary of matrix-dataframes, organized by config and testing fold.
    """
    # Get the mode of ALL the prediction shapes
    shapes_mode = []
    for config in matrices:
        for test_fold in matrices[config]:
            if not is_outer:
                for val_fold in matrices[config][test_fold]:
                    shapes_mode.append(shapes[config][test_fold][val_fold])
            else:
                shapes_mode.append(shapes[config][test_fold])
    shapes_mode = mode(shapes_mode)

    # Remove matrices whose prediction value length do not match the mode
    for config in matrices:
        for test_fold in matrices[config]:
            test_fold_matrices = []
            test_fold_matrices = []
            
            # Each testing fold will have an array of coresponding validation matrices
            if not is_outer:
                for val_fold in matrices[config][test_fold]:
                    val_fold_shape = shapes[config][test_fold][val_fold]
                    if val_fold_shape == shapes_mode:
                        test_fold_matrices.append(matrices[config][test_fold][val_fold])
                    else:
                        print(colored(
                            f"Warning: Not including the validation fold {val_fold} in the mean of ({config}, {test_fold})." +
                            f"\n\tThe predictions expected to have {shapes_mode} rows, but got {val_fold_shape} rows.\n",
                            "yellow"))
                matrices[config][test_fold] = test_fold_matrices
            else:
                test_fold_shape = shapes[config][test_fold]
                if test_fold_shape == shapes_mode:
                    test_fold_matrices.append(matrices[config][test_fold])
                    matrices[config][test_fold] = test_fold_matrices
                    test_fold_matrices.append(matrices[config][test_fold])
                    matrices[config][test_fold] = test_fold_matrices
    return matrices


def perform_and_output_matrix(valid_matrices, all_matrices, labels, output_path, is_outer, config, round_to, test_fold=None):
    # TODO 
    # Assign parameters and fix code
        # Check if length is valid for finding mean/stderr
        
    n_items_valid = len(valid_matrices)
    n_items_all = len(all_matrices)

    if not n_items_valid > 1:
        print(colored(f"Warning: Mean calculation skipped for testing fold {test_fold} in {config}."
                        + " Must have multiple validation folds.\n", 'yellow'))
        return

    index_labels = [["Truth"] * len(labels), labels]
    columns_labels = [["Predicted"] * len(labels), labels]

    # The mean of confusion matrices and the weighted matrices
    if type(labels) == dict:
        labels = list(labels.keys())
        
    matrix_avg = pd.DataFrame(0.0, columns=labels, index=labels)
    matrix_perc_avg = pd.DataFrame(0.0, columns=labels, index=labels)
    matrix_sum_temp = pd.DataFrame(0.0, columns=labels, index=labels)
    matrix_avg.index = matrix_perc_avg.index = matrix_sum_temp.index = index_labels
    matrix_avg.columns = matrix_perc_avg.columns = matrix_sum_temp.columns = columns_labels

    # See if more than one item to average
    b_weight = len(all_matrices) > 1

    # Percentage matrix values
    perc_values = {}

    # Absolute version
    # Loop through every validation fold and sum the totals and weighted totals
    for val_index in range(len(valid_matrices)):
        val_fold = pd.read_csv(valid_matrices[val_index], index_col=[0,1], header=[0,1])
        for row in labels:
            # Count the row total and add column-row items to total
            row_total = 0
            for col in labels:
                item = val_fold["Predicted"][col]["Truth"][row]
                matrix_avg["Predicted"][col]["Truth"][row] += item
                row_total += item
            
    # Percentage version
    # Loop through every validation fold and sum the totals and weighted totals
    for val_index in range(len(all_matrices)):
        val_fold = pd.read_csv(all_matrices[val_index], index_col=[0,1], header=[0,1])
        if b_weight:
            perc_values[val_index] = pd.DataFrame(0, columns=labels, index=labels)
            perc_values[val_index].index = index_labels
            perc_values[val_index].columns = columns_labels
        for row in labels:
            # Count the row total and add column-row items to total
            row_total = 0
            for col in labels:
                item = val_fold["Predicted"][col]["Truth"][row]
                matrix_sum_temp["Predicted"][col]["Truth"][row] += item
                row_total += item

            # Add to the weighted total
            for col in labels:
                item = val_fold["Predicted"][col]["Truth"][row]
                if row_total != 0:
                    weighted_item = item / row_total
                else:
                    weighted_item = 0
                perc_values[val_index]["Predicted"][col]["Truth"][row] = weighted_item
                matrix_perc_avg["Predicted"][col]["Truth"][row] += weighted_item

    # Divide the mean-sums by the length
    for row in labels:
        for col in labels:
            matrix_avg["Predicted"][col]["Truth"][row] /= n_items_valid
            matrix_perc_avg["Predicted"][col]["Truth"][row] /= n_items_all

    # The standard error of the mean and weighted mean
    matrix_err = pd.DataFrame(0.0, columns=labels, index=labels)
    matrix_perc_err = pd.DataFrame(0.0, columns=labels, index=labels)
    matrix_err.index = matrix_perc_err.index = index_labels
    matrix_err.columns = matrix_perc_err.columns = columns_labels
    
    # Average
    # Sum the differences between the true and mean values squared. Divide by the num matrices minus one.
    for val_index in range(len(valid_matrices)):
        val_fold = pd.read_csv(valid_matrices[val_index], index_col=[0,1], header=[0,1])
        for row in labels:
            for col in labels:
                
                # Sum the (true - mean) ^ 2
                true = val_fold["Predicted"][col]["Truth"][row]
                mean = matrix_avg["Predicted"][col]["Truth"][row]
                matrix_err["Predicted"][col]["Truth"][row] += (true - mean) ** 2

    # Percentage
    # Sum the differences between the true and mean values squared. Divide by the num matrices minus one.
    for val_index in range(len(all_matrices)):
        val_fold = pd.read_csv(all_matrices[val_index], index_col=[0,1], header=[0,1])
        for row in labels:
            for col in labels:
                
                # Sum the (true_weighted - mean_weighted) ^ 2
                true_weighted = perc_values[val_index]["Predicted"][col]["Truth"][row]
                mean_weighted = matrix_perc_avg["Predicted"][col]["Truth"][row]
                matrix_perc_err["Predicted"][col]["Truth"][row] += (true_weighted - mean_weighted) ** 2

    # Get the standard error
    for row in labels:
        for col in labels:
            if n_items_valid > 1:
                
                # Divide by N-1
                matrix_err["Predicted"][col]["Truth"][row] /= n_items_valid - 1
                # Sqrt the entire calculation
                matrix_err["Predicted"][col]["Truth"][row] = math.sqrt(matrix_err["Predicted"][col]["Truth"][row])
                # Divide by sqrt N
                matrix_err["Predicted"][col]["Truth"][row] /= math.sqrt(n_items_valid)
                
            if n_items_all > 1:
                
                # Divide by N-1
                matrix_perc_err["Predicted"][col]["Truth"][row] /= n_items_all - 1
                # Sqrt the entire calculation
                matrix_perc_err["Predicted"][col]["Truth"][row] = math.sqrt(matrix_perc_err["Predicted"][col]["Truth"][row])
                # Divide by sqrt N
                matrix_perc_err["Predicted"][col]["Truth"][row] /= math.sqrt(n_items_all)
            
                
                
    # Create a combination of the mean and std error
    matrix_combo = pd.DataFrame('', columns=labels, index=labels)
    matrix_perc_combo = pd.DataFrame(0, columns=labels, index=labels)
    matrix_combo.index = matrix_perc_combo.index = index_labels
    matrix_combo.columns = matrix_perc_combo.columns = columns_labels
    for row in labels:
        for col in labels:
            matrix_combo.loc[("Truth", row), ("Predicted", col)] = \
                f'{round(matrix_avg["Predicted"][col]["Truth"][row], round_to)} ± {round(matrix_err["Predicted"][col]["Truth"][row], round_to)}'
            if not matrix_perc_avg.empty:
                matrix_perc_combo.loc[("Truth", row), ("Predicted", col)] = \
                    f'{round(matrix_perc_avg["Predicted"][col]["Truth"][row], round_to)} ± {round(matrix_perc_err["Predicted"][col]["Truth"][row], round_to)}'
    
    # Output all of the mean and error dataframes
    if is_outer:
        output_folder = os.path.join(output_path)
        prefix = f"{config}"
    else:
        output_folder = os.path.join(output_path, f'{config}_{test_fold}/')
        prefix = f"{config}_{test_fold}"

    if not os.path.exists(output_folder): os.makedirs(output_folder)
            
    matrix_avg.round(round_to).to_csv(os.path.join(
        output_folder, f'{prefix}_conf_matrix_mean.csv'))          
    matrix_err.round(round_to).to_csv(os.path.join(
        output_folder, f'{prefix}_conf_matrix_stderr.csv'))
    matrix_combo.to_csv(os.path.join(
        output_folder, f'{prefix}_conf_matrix_mean_stderr.csv'))

    if not matrix_perc_avg.empty:
        matrix_perc_avg.round(round_to).to_csv(os.path.join(
            output_folder, f'{prefix}_conf_matrix_perc_mean.csv'))
        matrix_perc_err.round(round_to).to_csv(os.path.join(
            output_folder, f'{prefix}_conf_matrix_perc_stderr.csv'))
        matrix_perc_combo.to_csv(os.path.join(
            output_folder, f'{prefix}_conf_matrix_perc_mean_stderr.csv'))
 
    if is_outer:
        print(colored(f"Mean confusion matrix results created for {config}", 'green'))
    else:
        print(colored(f"Mean confusion matrix results created for {test_fold} in {config}", 'green'))

def get_mean_matrices(matrices, shapes, output_path, labels, round_to, is_outer):
    """ This function gets the mean confusion matrix of every inner loop.

    Args:
        matrices (dict): A dictionary of matrices organized by config and testing fold.
        shapes (dict): A dictionary of shapes organized by config and testing fold.
        output_path(string): The path to write the average matrices to.
        labels(list(str)): The labels of the matrix data.
        is_outer (bool): Whether this data is of the outer loop.
        is_outer (bool): Whether this data is of the outer loop.
    """
    if type(labels) == dict:
        labels = list(labels.keys())
    
    # Check that the output folder exists.
    if not os.path.exists(output_path): os.makedirs(output_path)

    # Get the valid matrices
    all_valid_matrices = get_matrices_of_mode_shape(shapes, matrices, is_outer)

    # The names of the rows/columns of every output matrix


    # Get the mean of each test fold + configuration
    for config in matrices:       
        if is_outer:
            valid_matrices = [path_confusion_matrix[0] for path_confusion_matrix in all_valid_matrices[config].values()]
            all_matrices = [path_confusion_matrix[0] for path_confusion_matrix  in matrices[config].values()]      
            perform_and_output_matrix(valid_matrices, all_matrices, labels, output_path, is_outer, config, round_to, test_fold=None)
        else:
            for test_fold in matrices[config]:
                valid_matrices = all_valid_matrices[config][test_fold]
                all_matrices = matrices[config][test_fold]
                perform_and_output_matrix(valid_matrices, all_matrices, labels, output_path, is_outer, config, round_to, test_fold=test_fold)

def main():
    """ The Main Program. """
    # Get program configuration and run using its contents
    config = parse_json(os.path.abspath('./results_processing/confusion_matrix/confusion_matrix_many_means_config.json'))

    # Read in the matrices to average
    matrices, shapes = get_input_matrices(config['matrices_path'], config['is_outer'])

    # Average the matrices
    get_mean_matrices(matrices, shapes, config['means_output_path'], config['label_types'], config['round_to'], config['is_outer'])


if __name__ == "__main__":
    """ Executes Program. """
    main()