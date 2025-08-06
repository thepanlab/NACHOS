# **Training**

<ul> 
    This module contains many various scripts that help process data from the data and training results. This can help to determine various performance metrics of the model. It does not directly affect the training process whatsoever. 
    All of these submodules can be run manually, or by using the provided Makefile. View the makefile or its documentation for more details.

</ul> <hr> <br>


+ ## ***Inner Loop***
    <ul> 
        This runs the inner loop of the k-fold cross validation process. This can be called using the Makefile and manually within the command line from within the scripts folder. Do to its modularization implementation, it will most likely break when called outside of this location. The default configuration folder is "training/training_config_files".
    </ul>

    ***Example:*** 
    >python3 -m training.training_sequential.loop_inner.training_loop_inner --file myfile.json ***OR*** --folder /myfolder
    
    <br>
    
    >mpirun -n 3 python3 -m training.training_multiprocessing.loop_inner.multiprocessed_training_loop_inner --file myfile.json ***OR*** --folder /myfolder

    <details>

    * ***Input:*** The configuration file or folder. *(Optional)*
    * ***Output:*** Trained models, prediction results, and other various metrics.
    * ***example_model_config.json:***
        ```json
        {
            "hyperparameters": {
                "batch_size": 24,
                "channels": 1,
                "cropping_position": [40, 10],
                "decay": 0.01,
                "do_cropping": true,
                "epochs": 6,
                "learning_rate": 0.01,
                "momentum": 0.9,
                "bool_nesterov": true,
                "patience": 10
            },

            "data_input_directory": "[path]/ep_intensity",
            "output_path": "[path]/training",
            "job_name": "InceptionV3_test_1",
            
            "k_epoch_checkpoint_frequency": 2,

            "shuffle_the_images": true,
            "shuffle_the_folds": false,
            "seed": 9,

            "class_names": ["fat", "ligament", "flavum", "epidural space", "spinal cord"],
            "selected_model_name": "InceptionV3",
            "subject_list": ["e1", "e2", "e3", "e4", "e5", "e6"],
            "test_subjects": ["e1", "e2"],
            "validation_subjects": ["e3", "e4"],

            "target_height": 241,
            "target_width": 181
        }

        ```
        * ***hyperparameters:*** These are the parameters needed for the training step.
          * ***batch_size:*** This will divide the input datasets into n training batches.
          * ***channels:*** The image channels in which the image will be processed on.
          * ***cropping_position:*** The position at which to crop the image.
          * ***decay:*** Decays the learning rate over time.
          * ***do_cropping:*** Whether to crop the input images.
          * ***epochs:*** The number of training epochs.
          * ***learning_rate:*** The learning speed.
          * ***momentum:*** Helps the learning rate's speed by speeding up the gradient descent search.
          * ***"bool_nesterov"***: Whether to apply Nesterov momentum
          * ***patience:*** How long to wait for improvement within early stopping.
        * ***data_input_directory:*** Where the input images are located. No specific structure is needed.
        * ***output_path:*** Where to write the results to.
        * ***job_name:*** The name of your job. Will mainly affect checkpointing file names.
        * ***k_epoch_checkpoint_frequency:*** How many epochs should checkpoints be saved.
        * ***shuffle_the_images:*** Whether to randomly shuffle the image paths.
        * ***shuffle_the_folds:*** Whether to randomly shuffle the training folds.
        * ***seed:*** The random seed.
        * ***class_names:*** The names of the image classes or labels.
        * ***selected_model_name:*** The model type to create. The choices are: resnet_50, resnet_VGG16, InceptionV3, ResNet50V2, and Xception.
        * ***subject_list:*** The list of all training subjects found in the data. 
        * ***test_subjects:*** The list of the particular testing subjects.
        * ***validation_subjects:*** The list of the particular validation subjects.
        * ***target_height:*** The target image height.
        * ***target_width:*** The target image width.
  
    </details> </hr> <br> <br>
<hr>

+ ## ***Distributed in multiple servers***
Server 1: 10.244.244.44 (3 processes: master process + 2 gpu process)
Server 2: 10.244.244.45 (1 process: 1 gpu process)

```bash
mpirun -n 4 --hosts 10.244.244.44,10.244.244.45:1 python3 -m training.training_multiprocessing.loop_inner.multiprocessed_training_inner_loop --file config_ngpu3.json
```

+ ## ***Outer Loop***
    <ul> 
        This runs the outer loop of the k-fold cross validation process. The only difference between running the inner and outer loop is its purpose, configuration, and names. They share most of their logic.
    </ul>

    ***Example:*** 
    >python3 -m training.training_sequential.loop_outer.training_outer_loop --file myfile.json ***OR*** --folder /myfolder
    
    <br>
    
    >mpirun -n 3 python3 -m training.training_multiprocessing.loop_outer.multiprocessed_training_outer_loop --file myfile.json ***OR*** --folder /myfolder

    <details>

    * ***Input:*** The configuration file or folder. *(Optional)*
    * ***Output:*** Trained models, prediction results, and other various metrics.
    * ***example_model_config.json:***
        ```json
        {
            "hyperparameters": {
                "batch_size": 24,
                "channels": 1,
                "cropping_position": [40, 10],
                "decay": 0.01,
                "do_cropping": true,
                "epochs": [4,6],
                "learning_rate": 0.01,
                "momentum": 0.9
            },

            "data_input_directory": "[path]/ep_intensity",
            "output_path": "[path]/training",
            "job_name": "InceptionV3_test_1",
            
            "k_epoch_checkpoint_frequency": 2,

            "shuffle_the_images": true,
            "shuffle_the_folds": false,
            "seed": 9,

            "class_names": ["fat", "ligament", "flavum", "epidural space", "spinal cord"],
            "selected_model_name": "InceptionV3",
            "subject_list": ["e1", "e2", "e3", "e4", "e5", "e6"],
            "test_subjects": ["e1", "e2"],

            "target_height": 241,
            "target_width": 181
        }

        ```
        * ***hyperparameters:*** These are the parameters needed for the training step.
          * ***batch_size:*** This will divide the input datasets into n training batches.
          * ***channels:*** The image channels in which the image will be processed on.
          * ***cropping_position:*** The position at which to crop the image.
          * ***decay:*** Decays the learning rate over time.
          * ***do_cropping:*** Whether to crop the input images.
          * ***epochs:*** The number of training epochs. You can specify a single value or a list of values according to each subject test.
          * ***learning_rate:*** The learning speed.
          * ***momentum:*** Helps the learning rate's speed by speeding up the gradient descent search.
        * ***data_input_directory:*** Where the input images are located. No specific structure is needed.
        * ***output_path:*** Where to write the results to.
        * ***job_name:*** The name of your job. Will mainly affect checkpointing file names.
        * ***k_epoch_checkpoint_frequency:*** How many epochs should checkpoints be saved.
        * ***shuffle_the_images:*** Whether to randomly shuffle the image paths.
        * ***shuffle_the_folds:*** Whether to randomly shuffle the training folds.
        * ***seed:*** The random seed.
        * ***class_names:*** The names of the image classes or labels.
        * ***selected_model_name:*** The model type to create. The choices are: resnet_50, resnet_VGG16, InceptionV3, ResNet50V2, and Xception.
        * ***subject_list:*** The list of all training subjects found in the data. 
        * ***test_subjects:*** The list of the particular testing subjects.
        * ***target_height:*** The target image height.
        * ***target_width:*** The target image width.
  
    </details> </hr> <br> <br>
<hr>

+ ## ***Random Search***
> python3 -m training.random_search.create_random_json -j myfile.json OR --json myfile.json OR --load_json myfile.json

<details>
* ***example_model_config.json:***:

```json
{
    "seed": 1234,

    "n_trials": 9,

    "hyperparameters": {
        "batch_size_min": 16,
        "batch_size_max": 128,
        "channels": 1,
        "cropping_position": [40, 10],
        "do_cropping": false,
        "epochs": 50,
        "learning_rate_min": 0.0001,
        "learning_rate_max": 0.01,
        "l_momentum": [0.5, 0.9, 0.99],
        "l_nesterov": [true, false],
        "l_models":["resnet_50", "InceptionV3",
                    "Xception" ],
        "patience": 20
    },

    "configurations_directory": "/home/pcallec/mif_outer/results/random_search_configurations/split1_random",
    "data_input_directory": "/home/pcallec/analyze_images/results/data/OCT_paper/split_random",
    "output_path": "/home/pcallec/mif_outer/results/random_search_results/split1",
    "job_name": "oct_split1",
    
    "k_epoch_checkpoint_frequency": 5,
    
    "shuffle_the_images": true,
    "shuffle_the_folds": false,
    
    "class_names": ["cortex", "medulla","pelvis"],
    
    "subject_list": ["fold1", "fold2", "fold3", "fold4", "fold5",
                    "fold6", "fold7", "fold8", "fold9", "fold10"],
    "test_subjects": ["fold1", "fold2", "fold3", "fold4", "fold5",
                    "fold6", "fold7", "fold8", "fold9", "fold10"],
    "validation_subjects": ["fold1", "fold2", "fold3", "fold4", "fold5",
                            "fold6", "fold7", "fold8", "fold9", "fold10"],
    
    "image_size": [210, 185],
    "target_height": 210,
    "target_width": 185
}
```

- **`seed`**: Random seed used for reproducibility.  
- **`n_trials`**: Number of random hyperparameter configurations to generate and evaluate.

- **`batch_size_min`**, **`batch_size_max`**: Range of batch sizes to sample from (powers of 2, e.g., 16 to 128).
- **`channels`**: Number of image channels (e.g., 1 for grayscale).
- **`cropping_position`**: `[row, col]` coordinate to start cropping (if cropping is enabled).
- **`do_cropping`**: Whether to crop the input images.
- **`epochs`**: Number of training epochs.
- **`learning_rate_min`**, **`learning_rate_max`**: Range of learning rates to sample from (e.g., 0.0001 to 0.01).
- **`l_momentum`**: List of momentum values for the SGD optimizer (e.g., `[0.5, 0.9, 0.99]`).
- **`l_nesterov`**: Whether to apply Nesterov momentum (e.g., `[true, false]`).
- **`l_models`**: Model architectures to choose from:
  - `"resnet_50"`
  - `"InceptionV3"`
  - `"Xception"`
- **`patience`**: Number of epochs to wait for improvement before applying early stopping.

- **`data_input_directory`**: Path to the input dataset (no specific structure required).
- **`output_path`**: Path to save the model outputs and results.
- **`configurations_directory`**: Directory to save the generated hyperparameter configurations.
- **`job_name`**: Name of the job used in logging and file naming.
- **`k_epoch_checkpoint_frequency`**: How frequently (in epochs) to save model checkpoints.

- **`shuffle_the_images`**: Whether to randomly shuffle image paths before training.
- **`shuffle_the_folds`**: Whether to randomly shuffle the assignment of folds.

- **`class_names`**: List of class labels (e.g., `["cortex", "medulla", "pelvis"]`).
- **`subject_list`**: List of all subject folds (e.g., `["fold1", ..., "fold10"]`).
- **`test_subjects`**: Folds to be used for testing.
- **`validation_subjects`**: Folds to be used for validation.

- **`image_size`**: Size of each image `[height, width]`.
- **`target_height`**, **`target_width`**: Dimensions that images are resized to before training.
  
Output configurations can be found at [`results/random_search_configurations/algorithm1_Xray_split1_random`](/results/random_search_configurations/algorithm1_Xray_split1_random)
    </details> </hr> <br> <br>
<hr>

+ ## ***Checkpoint and Logging Modules***
    <ul> 
        This module contains various scripts that aid in checkpointing the model and continuing from previous training data.
    </ul> <br>
    <details>
    <summary>Show/Hide files</summary>

    1) ### ***checkpointer.py:***
        <ul> 
            This has the ability to write and load checkpoints. This allows the model to continue off from a previous training session. The checkpoints are saved in the same format as regular models, in h5 form.
        </ul>

    2) ### ***logger.py:***
        <ul> 
            This allows the training loop to log the most recent training state. This will allow the training loop to carry off from a cancelled job. Things like the testing subject, validation subject, and various training fold properties are stored using the functions within. It has the ability to read, write, and delete log files.
        </ul>

    </details> <br> <br>
<hr>



+ ## ***Training Modules***
    <ul> 
        This module contains most of the main training loop functions.
    </ul> <br>
    <details>
    <summary>Show/Hide files</summary>

    1) ### ***Data Processing: fold_generator.py:***
        <ul> 
            This will generate a list of training folds and the number of rotations for a given subject.
        </ul>

    2) ### ***Data Processing: index_getter.py:***
        <ul> 
            This will generate ordered lists for the labels, label-indexes, and subjects of the input images from their names.
        </ul>

    3) ### ***Data Processing: training_preparation.py:***
        <ul> 
            This is where basic training loop data is generated. Things like files, indexes, and folds are created here.
        </ul>

    4) ### ***Image Processing: image_getter.py:***
        <ul> 
            This retrieves every image from the given input file path.
        </ul>

    5) ### ***Image Processing: image_parser.py:***
        <ul> 
            This parses the input images as tensors.
        </ul>

    6) ### ***Image Processing: image_reader.py:***
        <ul> 
            A (currently unused) bit of code. Meant to implememt custom image reading classes.
        </ul>

    7) ### ***Model Processing:model_creator.py:***
        <ul> 
            This generates a model object, based on the given configuration.
        </ul>

    8) ### ***Output Processing: console_printing.py:***
        <ul> 
            This contains a basic printing function. It may be removed later.
        </ul>

    9) ### ***Output Processing: result_outputter.py:***
        <ul> 
            This outputs various training metrics after the process is done within each fold.
        </ul>

    10) ### ***Training Processing: training_fold.py:***
        <ul> 
            This is the main training function. Here is where the model is trained and its data is saved within a log and checkpoint.
        </ul>

    11) ### ***Training Processing: ttraining_loop.py:***
        <ul> 
            This module runs all of the training folds for a particular subject.
        </ul>

    </details> <br> <br>
<hr>


