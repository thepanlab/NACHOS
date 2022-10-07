from util.get_config import parse_json
from termcolor import colored
from util import path_getter
import pandas as pd
import math
import os


def count_epochs(history_paths):
    """ Creates a csv file for every model's history """
    # Store output dataframes by model
    model_dfs = {}

    # Each model will have its own dictionary, a later dataframe
    for model in history_paths:
        model_dfs[model] = {}

        # Every subject is the kth test set (row), find the column values
        for row in history_paths[model]:
            row_name = row
            model_dfs[model][row_name] = {}

            # Read in the data at this path
            for path in history_paths[model][row]:
                # Check if the fold has files to read from
                if not path:
                    raise Exception(colored("Error: model '" + model + "' and target '" + target_id
                                            + "' had no history file detected.", 'red'))

                # Read the file for this column/subject, get number of rows (epochs)
                data = pd.read_csv(path)
                min_index = -1
                min_epoch = float("inf")
                for row in data['val_loss'].index:
                    if data['val_loss'][row] < min_epoch:
                        min_epoch = data['val_loss'][row]
                        min_index = row

                # Add the epoch with the lowest loss the model's dataframe 
                col_name = path.split("/")[-2].split("_")[-1]
                model_dfs[model][row_name][col_name] = min_index

    # Return a dictionary of counts
    return model_dfs


def print_counts(epochs, output_path, config_nums):
    """ This will output a CSV of the epoch-counts """
    # Create a new dataframe to output
    col_names = ["test_fold", "config", "config_index", "val_fold", "epochs"]
    df = pd.DataFrame(columns=col_names)

    # Re-format data to match the columns above
    configs = list(epochs.keys())
    for testing_fold_index in range(len(epochs[configs[0]])):
        for config in configs:
            testing_fold = list(epochs[config].keys())[testing_fold_index]
            for validation_fold in epochs[config][testing_fold]:

                # Each row should contain the given columns
                df = df.append({
                    col_names[0]: testing_fold,
                    col_names[1]: config,
                    col_names[2]: config_nums[config],
                    col_names[3]: validation_fold,
                    col_names[4]: epochs[config][testing_fold][validation_fold]
                }, ignore_index=True)

    # Print to file
    file_name = 'epochs.csv'
    df = df.sort_values(by=[col_names[0], col_names[2], col_names[1]], ascending=True)
    df.to_csv(os.path.join(output_path, file_name), index=False)
    print(colored('Successfully printed epoch results to: ' + file_name, 'green'))


def print_stderr(epochs, output_path, config_nums):
    """ This will output a CSV of the average epoch standard errors """
    # Create a new dataframe to output
    col_names = ["test_fold", "config", "config-index", "avg_epochs", "std_err"]
    df = pd.DataFrame(columns=col_names)

    # Re-format data to match the columns above
    for config in epochs:
        for test_fold in epochs[config]:

            # Count epochs, get mean
            epoch_mean = 0
            n_val_folds = len(epochs[config][test_fold])
            for validation_fold in epochs[config][test_fold]:
                epoch_mean += epochs[config][test_fold][validation_fold]
            epoch_mean = epoch_mean / n_val_folds

            #  Get standard deviation
            stdev = 0
            for validation_fold in epochs[config][test_fold]:
                stdev += (epochs[config][test_fold][validation_fold] - epoch_mean) ** 2
            stdev = math.sqrt(stdev / (n_val_folds - 1))

            # Each row should contain the given columns
            df = df.append({
                col_names[0]: test_fold,
                col_names[1]: config,
                col_names[2]: config_nums[config],
                col_names[3]: epoch_mean,
                col_names[4]: stdev / math.sqrt(n_val_folds)
            }, ignore_index=True)

    # Print to file
    file_name = 'epoch_avg_and_stderr.csv'
    df = df.sort_values(by=[col_names[0], col_names[1]], ascending=True)
    df.to_csv(os.path.join(output_path, file_name), index=False)
    print(colored('Successfully printed epoch averages/stderrs to: ' + file_name, 'green'))


def main(config=None):
    """ The main body of the program """
    # Obtain a dictionary of configurations
    if config is None:
        config = parse_json('epoch_counting_config.json')
    if not os.path.exists(config['output_path']):
        os.makedirs(config['output_path'])

    # Get the necessary input files
    history_paths = path_getter.get_history_paths(config['data_path'])

    # Count the number of epochs within every file
    epochs = count_epochs(history_paths)

    # Get config nums (E.g config 1)
    config_nums = path_getter.get_config_indexes(config['data_path'])

    # Output the counts
    print_counts(epochs, config['output_path'], config_nums)

    # Output the stderr
    print_stderr(epochs, config['output_path'], config_nums)


if __name__ == "__main__":
    main()
