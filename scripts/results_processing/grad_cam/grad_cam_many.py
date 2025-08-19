from training.training_modules.image_processing.image_getter import get_files
from training.training_modules.image_processing.image_getter import get_files
from results_processing.grad_cam import grad_cam
from termcolor import colored
from util import get_config
import pandas as pd
import numpy as np
import os


def filter_file_list(path_list, query):
    """ Filters a list of files.

    Args:
        path_list (list): A list of image paths.
        query (dict): A dictionary of values linked to the image name. Items should be empty arrays if not used.

    Returns:
        list: A list of files to process.
    """
    for item in ["test_subject", "true_label"]:
        if query[item]:
            for i in query[item]:
                path_list = [path for path in path_list if i in path.split('/')[-1]]
    if query["cutoff_number_of_results"] > 0:
        path_list = path_list[:query["cutoff_number_of_results"]]
    if query["sort_images"]:
        path_list.sort()
    return path_list


def filter_csv(csv, query):
    """ Filters a Pandas dataframe by some given query conditions.

    Args:
        csv (pandas.DataFrame): A dataframe of tabled prediction info.
        query (dict): A dictionary of values linked to the csv. Items should be empty arrays if not used.

    Returns:
        dict: A dictionary of paths separated by subject, true labels, correctness, and predicted labels
    """
    # Filter the items that directly correlate to the csv values
    for item in ['test_subject', 'match', 'true_label', 'pred_label', 'true_label_index', 'pred_label_index']:
        if query[item]:
            csv = csv[csv[item].isin(query[item])]
            
    # Filter the true and predicted pairs, if given
    pair_items = {
        'true_predicted_label_pairs': ['true_label', 'pred_label'], 
        'true_predicted_index_pairs': ['true_label_index', 'pred_label_index']
    }
    for item in pair_items:
        new_df = pd.DataFrame(columns=csv.columns)
        if query[item]:
            for pair in query[item]:
                new_df = pd.concat([new_df, csv[ (csv[pair_items[item][0]]==pair[0]) & (csv[pair_items[item][1]]==pair[1]) ]])
            csv = new_df
            
    # If needed, sort and limit the files/rows
    if query["cutoff_number_of_results"] > 0:
        csv = csv.head(query["cutoff_number_of_results"])
    if query["sort_images"]:
        csv = csv.sort_values(by=['filename'])
    
    # Separate files by test subject
    path_dict = {}
    for subject in np.unique(np.array(csv['test_subject'].tolist())):
        path_dict[subject] = {}
        subject_df = csv[csv['test_subject'] == subject]
        
        # Separate files by label
        for true_label in np.unique(np.array(subject_df['true_label'].tolist())):
            path_dict[subject][true_label] = {}
            true_label_df = subject_df[subject_df['true_label'] == true_label]
        
            # Separate files by correctness
            path_dict[subject][true_label]['correct'] = true_label_df[true_label_df['match'] == True]['filepath'].tolist()
        
            # Separate files by incorrectness
            path_dict[subject][true_label]['incorrect'] = {}
            incorrect_df = true_label_df[true_label_df['match'] == False]
            for pred_label in np.unique(np.array(incorrect_df['pred_label'].tolist())):
                path_dict[subject][true_label]['incorrect'][pred_label] = incorrect_df[incorrect_df['pred_label'] == pred_label]['filepath'].tolist()
    
    # Return the image paths as a list
    return path_dict
        
             
def filter_images(input_path, query):
    """ Filters the input images by the given query.

    Args:
        input_path (str): Path to a folder or tabled prediction info CSV.
        query (dict): The given query from the config.

    Returns:
        dict or list: A collection of image file paths.
    """
    # If directory, read in and filter the image paths.
    if os.path.isdir(input_path):
        print(colored("Note: An input directory of images was given. Only subjects and true labels can be filtered.", 'yellow'))
        image_paths = get_files(input_path, False, 1)
        image_paths = get_files(input_path, False, 1)
        return filter_file_list(image_paths, query)
    
    # Read in the tabled prediction info CSV, filter it, and return the image paths
    tabled_info = pd.read_csv(input_path)
    return filter_csv(tabled_info, query)


def generate_json_and_run(json_init, input_path, output_path):
    """ Generates the json needed to run the program, then calls the main function.

    Args:
        json_init (json): A base json used for all images.
        input_path (str): A path to the input image.
        output_path (str): A path to the output folder.

    """
    this_json = json_init.copy()
    this_json['input_img_address'] = input_path
    this_json['output_image_address'] = output_path
    grad_cam.main(this_json)


def run_program(image_addrs, config):
    """ Runs the main program for each image

    Args:
        image_addrs (list of str): Locations of all images.
        config (dict): This program's configuration.
    """
    # Get the layer to use for all items.
    last_conv_layer_name = grad_cam.get_layer_name(grad_cam.load_data(config['input_model_address']), config["last_conv_layer_name"])
    
    # All outputs will use the base items given in the configuration
    json_init = {
        'input_model_address': config['input_model_address'],
        'last_conv_layer_name': last_conv_layer_name,
        'alpha': config['alpha'],
        'index_class_gradcam':config['index_class_gradcam'] 
    }
    
    # If the addresses come in a list, output to the same directory
    if type(image_addrs) == list:
        for path in image_addrs:
            generate_json_and_run(json_init, path, config['output_directory'])
    
    # If they are in a dictionary, output based on dictionary structure 
    else:
        for subject in image_addrs:
            for true_label in image_addrs[subject]:
                
                # For every correctly predicted image, write to a single subdirectory
                for path in image_addrs[subject][true_label]['correct']:
                    output_path = os.path.join(
                        config['output_directory'],
                        f"subject_{subject}/{true_label}_correct"
                    )
                    generate_json_and_run(json_init, path, output_path)
                    generate_json_and_run(json_init, path, output_path)
                    
                # Else, separate incorrect images by the predicted label
                for pred_label in image_addrs[subject][true_label]['incorrect']:
                    for path in image_addrs[subject][true_label]['incorrect'][pred_label]:
                        output_path = os.path.join(
                            config['output_directory'],
                            f"subject_{subject}/{true_label}_incorrect_labels/{pred_label}"
                        )
                        generate_json_and_run(json_init, path, output_path)
                

def main(config=None):
    """ The main program.

    Args:
        config (dict, optional): A custom configuration. Defaults to None.
    """
    # Obtaining dictionary of configurations from the json file
    if config is None:
        config = get_config.parse_json('./results_processing/grad_cam/grad_cam_many_config.json')

    # Read in and filter the image paths to include only the relevant items
    img_addrs = filter_images(config["input_directory_or_tabled_info"], config["query"])
    
    # Run the program for each image address
    run_program(img_addrs, config)
    print(colored("Finished processing all images.", 'magenta'))


if __name__ == "__main__":
    """ Executes the program """
    main()