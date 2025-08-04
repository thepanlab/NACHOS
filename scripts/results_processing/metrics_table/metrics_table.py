from sklearn.metrics import accuracy_score
from util.get_config import parse_json
from scipy.stats import sem as std_err
from termcolor import colored
from util import path_getter
import pandas as pd
import os


def read_data(paths):
    """ This will read in file-data into a config-subject dictionary

    Args:
        paths (dict): A dictionary of paths sorted by config and test fold.

    Raises:
        Exception: If no data files are found within some config and test fold.

    Returns:
        dict: A dictionary of dataframes.
    """
    # Return this
    results = {}

    # Create a dictionary for each config
    for config in paths:
        results[config] = {}

        # Create an array for each target-id
        for test_fold in paths[config]:
            results[config][test_fold] = {}

            # Check if the fold has files to read from
            if not paths[config][test_fold]:
                raise Exception(colored("Error: config '" + config + "' and testing fold '"
                                        + test_fold + "' had no indexed CSV files detected. "
                                        + "Double-check the data files are there.", 'red'))

            # For each file, read and store it
            for validation_fold_path in paths[config][test_fold]:
                val_fold = validation_fold_path.split("/")[-3].split("_")[-1]
                results[config][test_fold][val_fold] = pd.read_csv(validation_fold_path, header=None).to_numpy()

    # Return the dictionary
    return results


def get_accuracies_and_stderr(true, pred):
    """ Gets the accuracies of each fold-config index.

    Args:
        true (dict): A dictionary of true values sorted by config and test fold.
        pred (dict): A dictionary of predicted values sorted by config and test fold.

    Returns:
        dict: Two dictionaries of relative and real accuracies and standard error. Sorted by config and test fold.
    """
    # Get a table of accuracy for every config
    accs = {'individual': {}, 'column': {}, 'row': {}}
    for config in pred:
        accs['individual'][config] = {}
        accs['column'][config] = {}
        accs['row'][config] = {}
        
        # Get the accuracy and other metrics of every test-val fold pair
        for test_fold in pred[config]:
            accs['individual'][config][test_fold] = {}
            accs['row'][config][test_fold] = [0]

            # Get the accuracy
            for val_fold in pred[config][test_fold]:
                pred_vals = pred[config][test_fold][val_fold]
                true_vals = true[config][test_fold][val_fold]
                
                # Test-val pair
                accs['individual'][config][test_fold][val_fold] = accuracy_score(y_true=true_vals, y_pred=pred_vals)
                
                # The row accuracy
                accs['row'][config][test_fold] += [accs['individual'][config][test_fold][val_fold]]
                
                # The column accuracy
                if val_fold not in accs['column'][config]:
                    accs['column'][config][val_fold] = [0]
                accs['column'][config][val_fold] += [accs['individual'][config][test_fold][val_fold]]
            
            # Get the average test fold (row) accuracy
            accs['row'][config][test_fold] = sum(accs['row'][config][test_fold]) / (len(accs['row'][config][test_fold])-1)
                
        # Get the average val fold (col) accuracies
        for val_fold in accs['column'][config]:
            accs['column'][config][val_fold] = sum(accs['column'][config][val_fold]) / (len(accs['column'][config][val_fold])-1)
            
    # Calculate the standard error of the accuracies
    errs = {}
    for config in pred:
        errs[config] = {}
        
        # Get the standard error of the test-fold accs
        for test_fold in pred[config]:
            errs[config][test_fold] = std_err(list(accs['individual'][config][test_fold].values()))
                
    # Return accuracy
    return accs, errs


def total_output(accuracies, standard_error, output_path, output_file, round_to, is_outer):
    """ Produces a table of accuracies and standard errors by config and fold

    Args:
        accuracies (dict): A dictionary of accuracies sorted by config and test fold.
        standard_error (dict): A dictionary of errors sorted by config and test fold.
        output_path (str): A string of the directory the output CSV should be written to.
        output_file (str): Name prefix of the output files.
        is_outer (bool): If the data is from the outer loop.
    """
    # Get names of columns and subjects
    configs = list(accuracies['individual'].keys())
    test_folds = list(accuracies['individual'][configs[0]].keys())
    
    # Extracting val_folds from the test fold values for accuracies
    val_folds_set = set()
    for dict_val_results in accuracies['individual'][configs[0]].values():
        val_folds_set.update(list(dict_val_results.keys()))
    
    val_folds = list(val_folds_set)
    test_folds.sort()
    val_folds.sort()
    
    # Get output path
    if is_outer:
        output_path = os.path.join(output_path, 'outer_loop')
    else:
        output_path = os.path.join(output_path, 'inner_loop')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    # Create a table for each config
    for config in configs:
        df = pd.DataFrame(columns=val_folds+['avg_acc', 'stderr'])
        
        # For every test fold, add column values
        for test_fold in test_folds:
            row = {}          
            
            # Add the individual accuracies
            for val_fold in accuracies['individual'][config][test_fold]:
                row[val_fold] = [accuracies['individual'][config][test_fold][val_fold]]
                
            # Add the mean and stderr
            row['avg_acc'] = [accuracies['row'][config][test_fold]]
            row['stderr'] = [standard_error[config][test_fold]]
            
            # Append the row to the dataframe
            df.loc[test_fold] = pd.DataFrame.from_dict(row).loc[0]
            
        # Add a row of val-fold accuracies
        row = {}
        for val_fold in val_folds:
            row[val_fold] = [accuracies['column'][config][val_fold]]
        df.loc['avg'] = pd.DataFrame.from_dict(row).loc[0]
            
        # Create and save the Pandas dataframe
        if is_outer:
            df.to_csv(f'{output_path}/{output_file}_{config}_outer.csv')
        else:
            df.to_csv(f'{output_path}/{output_file}_{config}_inner.csv')


def main(config=None):
    """ The main program.

    Args:
        config (dict): A JSON configuration as a dictionary. (Optional)
    """
    # Obtain a dictionary of configurations
    if config is None:
        config = parse_json('./results_processing/metrics_table/metrics_table_config.json')

    # Get the necessary input files
    true_paths = path_getter.get_subfolder_files(config['data_path'], "true_label", isIndex=True, getValidation=True, isOuter=config['is_outer'])
    pred_paths = path_getter.get_subfolder_files(config['data_path'], "prediction", isIndex=True, getValidation=True, isOuter=config['is_outer'])

    # Read in each file into a dictionary
    true = read_data(true_paths)
    pred = read_data(pred_paths)
    
    # Get accuracies and standard error
    accuracies, stderr = get_accuracies_and_stderr(true, pred)

    # Graph results
    total_output(accuracies, stderr, config['output_path'], config['output_filename'], config['round_to'], is_outer=config['is_outer'])
    print(colored("Finished writing the metric tables.", 'green'))


if __name__ == "__main__":
    """ Executes the program """
    main()