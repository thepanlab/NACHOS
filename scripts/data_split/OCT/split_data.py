import random
import pathlib
import shutil
import pandas as pd
import numpy as np

from src_util import util_reading

def split_randomly(config_json):
    # Get list of files 
    path_directory = pathlib.Path(config_json["directory_data"])
    
    suffix_file = config_json["type_file"]
    
    list_files = list(path_directory.glob(f'**/*.{suffix_file}'))
    list_files.sort()
    
    # get list for categories to split later evenly
    dict_indices = {}
    for category in config_json["list_classes"]:
        dict_indices[category] = []
        
    for index, file_path in enumerate(list_files):
        filename = file_path.name
        category = filename.split("_")[1]
        dict_indices[category].append(index)
        
    # get sublist
    random.seed(config_json["random_seed"])
    dict_categories_list = {}
    
    for i, category in enumerate(config_json["list_classes"]):
        random.seed( config_json["random_seed"]*(i+1) )
        dict_categories_list[category] = [ list_files[i] for i in dict_indices[category]]
        dict_categories_list[category] = random.sample(dict_categories_list[category], len(dict_categories_list[category]))

    # # Randomize then
    # list_files_randomized = random.sample(list_files, len(list_files))
       
    # Split according to number of folds
    # assuming number of images divide evenly
    multiplier = len(list_files)/(config_json["number_of_folds"]*len(config_json["list_classes"]))
    list_indices = [int(multiplier*i) for i in range(config_json["number_of_folds"]+1)]
    
    path_output_random = pathlib.Path(config_json["directory_output_random"])
    path_output_random.mkdir(parents=True, exist_ok=True)
    
    df_data = pd.DataFrame(columns = ["fold", "category",
                                      "original_path", "original_name",
                                      "new_path", "new_name"])

    for ite_fold in range(config_json["number_of_folds"]):
        
        for category in config_json["list_classes"]:
            l_temp_fold = dict_categories_list[category][list_indices[i]:list_indices[i+1]]
            l_temp_fold = random.sample(l_temp_fold, len(l_temp_fold))

            for index_file, file_path in enumerate(l_temp_fold):
            
                filename = file_path.name
                # extract class
                category = filename.split("_")[1]
                suffix = file_path.suffix
                
                new_name = f"{index_file+1}_fold{ite_fold+1}_{category}{suffix}"
                
                destination_directory = path_output_random / f"fold{ite_fold+1}" / category
                destination = destination_directory / new_name
                # destination = file_path.with_stem(path_output_random)
                destination_directory.mkdir(parents=True, exist_ok=True)
                
                destination.write_bytes(file_path.read_bytes())
                
                df_temp = pd.DataFrame({"original_path":[file_path],
                                        "original_name":[filename],
                                        "fold":[ite_fold+1],
                                        "category": [category],
                                        "new_path":[destination],
                                        "new_name":[new_name]})
            
                df_data = pd.concat([df_data, df_temp], ignore_index=True)

        path_df = path_output_random.parent/ f"df_conversion_random_{config_json['number_of_folds']}folds.csv"
        
        df_data.to_csv(path_df)

def split_volume(config_json):
    # Get list of files 
    path_directory = pathlib.Path(config_json["directory_data"])
    
    path_output_volume = pathlib.Path(config_json["directory_output_volume"])
    path_output_volume.mkdir(parents=True, exist_ok=True)
    
    suffix_file = config_json["type_file"]
    
    list_files = list(path_directory.glob(f'**/*.{suffix_file}'))
    list_files.sort()
    
    # get list for categories to split later evenly
    df_all = pd.DataFrame()   
       
    for index, file_path in enumerate(list_files):
        filename = file_path.name
        category = filename.split("_")[1]
        
        subject = filename.split("_")[0]
        volume = filename.split("_")[2]
        
        subject_volume = f"{subject}-{volume}"
        
        df_temp = pd.DataFrame({"filename":[filename],
                        "filepath":[file_path],
                        "category": [category],
                        "subject-volume":[subject_volume]})
    
        df_all = pd.concat([df_all, df_temp], ignore_index=True)
        
    a_subject_volume = df_all["subject-volume"].unique()
    
    np.random.seed(config_json["random_seed"])
    np.random.shuffle(a_subject_volume)
    
    multiplier = len(a_subject_volume)/(config_json["number_of_folds"])
    list_indices = [int(multiplier*i) for i in range(config_json["number_of_folds"]+1)]
    
    df_table_conversion = pd.DataFrame()
    
    for ite_fold in range(config_json["number_of_folds"]):
        a_subject_volume_subset = a_subject_volume[list_indices[ite_fold]:list_indices[ite_fold+1]]
        df_subset_temp = df_all[df_all["subject-volume"].isin(a_subject_volume_subset)].reset_index(drop=True)
        df_subset_temp.insert(0, "fold", ite_fold+1)
        
        # split by categories
        for category in config_json["list_classes"]:
            df_subset_temp_category = df_subset_temp.query(f"category == '{category}'").reset_index(drop=True)
        
            for index, row in df_subset_temp_category.iterrows():

                filename = row["filename"]
                file_path = row["filepath"]
                # extract class
                suffix = file_path.suffix
                
                new_name = f"{index+1}_fold{ite_fold+1}_{category}{suffix}"
                
                destination_directory = path_output_volume / f"fold{ite_fold+1}" / category
                destination = destination_directory / new_name
                # destination = file_path.with_stem(path_output_random)
                destination_directory.mkdir(parents=True, exist_ok=True)
                
                destination.write_bytes(file_path.read_bytes())
            
                df_subset_temp_category.loc[index, "new_name"] = new_name
                df_subset_temp_category.loc[index, "new_path"] = destination
                
            df_table_conversion = pd.concat([df_table_conversion,
                                            df_subset_temp_category], ignore_index=True)

        
    path_df = path_output_volume.parent/ f"df_conversion_volume_{config_json['number_of_folds']}folds.csv"
    df_table_conversion.to_csv(path_df)

def main():

    args = util_reading.get_parser()
    config_json = util_reading.get_json(args)

    split_randomly(config_json)
    split_volume(config_json)

if __name__ == "__main__":
    main()