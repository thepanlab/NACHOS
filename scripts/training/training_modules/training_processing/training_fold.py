from training.training_modules.output_processing.result_outputter import output_results
from training.training_modules.model_processing.model_creator import TrainingModel
from training.training_modules.image_processing.image_parser import *
from training.training_checkpointing_logging.checkpointer import *
from training.training_checkpointing_logging.logger import *
from termcolor import colored
from time import perf_counter
from tensorflow import keras
import tensorflow as tf
import fasteners
from pathlib import Path
import datetime


class _FoldTrainingInfo():
    def __init__(self, fold_index, config, testing_subject, rotation_subject, files, folds, indexes, label_position, n_epochs, rank=None, is_outer=False):
        """ Initializes a training fold info object.

        Args:
            fold_index (int): The fold index within the loop.
            config (dict): The training configuration.
            test_subject (str): The test subject name.
            rotation_subject (str): The rotation subject name.
            files (list of str): A list of filepaths to images.
            folds (list of dict): A list of fold partitions
            indexes (dict of lists): A list of true indexes.
            label_position (int): Location in the filename of the label.
            
            rank (int): An optional value of some MPI rank. Default is none. (Optional)
            is_outer (bool): If this is of the outer loop. Default is none. (Optional)
        """ 
        self.datasets = {
            'testing':    {'files': [], 'indexes': [], 'labels': [], 'ds': None},
            'training':   {'files': [], 'indexes': [], 'labels': [], 'ds': None},
            'validation': {'files': [], 'indexes': [], 'labels': [], 'ds': None}
        }
        self.testing_subject = testing_subject
        self.rotation_subject = rotation_subject
        
        self.rank = rank
        self.files = files
        self.folds = folds
        self.config = config
        self.indexes = indexes
        self.is_outer = is_outer
        self.fold_index = fold_index
        self.label_position = label_position
        self.n_epochs = n_epochs
        
        self.model = None
        self.callbacks = None
        self.checkpoint_name = None
        
        # If MPI, specify the job name by task
        if self.rank:
            self.config['job_name'] = f"{config['job_name']}_test_{testing_subject}" \
            if is_outer else f"{config['job_name']}_test_{testing_subject}_val_{rotation_subject}"
        
        
    def run_all_steps(self):
        """ This will run all of the steps in order to create the basic training information. """
        self.get_dataset_info()
        self.create_model()
        self.create_callbacks()
        
        
    def get_dataset_info(self):
        """ Get the basic data to create the datasets from.
            This includes the testing and rotation_subjects, file paths,
            label indexes, and labels.
        """
        # Loop through all image files and determine which dataset they belong to
        for index, file_path in enumerate(self.files):            
            dataset = ''
            subject_name = self.indexes['subjects'][index]
            
            # Outer loop
            if self.is_outer:
                if subject_name == self.testing_subject:
                    dataset = 'testing'
                elif subject_name in self.config['subject_list']:
                    dataset = 'training'
                else:
                    continue
                
            # Inner loop
            else:
                if subject_name == self.testing_subject:
                    dataset = 'testing'
                elif subject_name == self.rotation_subject:
                    dataset = 'validation'
                elif subject_name in self.config['subject_list']:
                    dataset = 'training'
                else:
                    continue
                
            # Append dataset item
            self.datasets[dataset]['files'].append(file_path)
            self.datasets[dataset]['indexes'].append(index)
            self.datasets[dataset]['labels'].append(self.indexes['labels'][index])
            
            
    def create_model(self):  
        """ Create the initial model for training. """
        self.model = TrainingModel(
            self.config['hyperparameters'],
            self.config['selected_model_name'], 
            self.config['target_height'], 
            self.config['target_width'], 
            self.config['class_names']
        )
            
    
    def create_callbacks(self):
        """ Create the training callbacks. 
            This includes early stopping and checkpoints.
        """
        # Get the job name for saving
        if self.is_outer:          
            self.checkpoint_prefix = f"{self.config['job_name']}_test_{self.testing_subject}_config_{self.config['selected_model_name']}"
        else:
            self.checkpoint_prefix = f"{self.config['job_name']}_test_{self.testing_subject}_val_{self.rotation_subject}_config_{self.config['selected_model_name']}"
           
            
        # Training checkpoints
        checkpoints = Checkpointer(
            self.n_epochs,
            self.config['k_epoch_checkpoint_frequency'], 
            self.checkpoint_prefix, 
            self.rank,
            os.path.join(self.config['output_path'], 'checkpoints')
        )
        
        step_lr_logger = StepLearningRateLogger()
        
        # Early stopping
        if not self.is_outer:
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=self.config['hyperparameters']['patience'], 
                restore_best_weights=True
            )
            self.callbacks = (step_lr_logger, early_stopping, checkpoints)
        else:
            self.callbacks = (step_lr_logger, checkpoints)
        


class Fold():
    def __init__(self, fold_index, config, testing_subject, rotation_subject, files, folds, indexes, label_position, n_epochs, rank=None, is_outer=False):
        """ Initializes a training fold object.

        Args:
            fold_index (int): The fold index within the loop.
            config (dict): The training configuration.
            test_subject (str): The test subject name.
            rotation_subject (str): The rotation_subject name.
            files (list of str): A list of filepaths to images.
            folds (list of dict): A list of fold partitionss
            indexes (dict of lists): A list of true indexes.
            label_position (int): Location in the filename of the label.
            
            rank (int): An optional value of some MPI rank. Default is none. (Optional)
            is_outer (bool): If this is of the outer loop. Default is none. (Optional)
        """ 
        self.fold_info = _FoldTrainingInfo(
            fold_index, 
            config, 
            testing_subject, 
            rotation_subject,
            files, 
            folds, 
            indexes, 
            label_position,
            n_epochs,
            rank,
            is_outer
        )
        self.history = None
        self.time_elapsed = None
        self.checkpoint_epoch = 0
        self.rank = rank
        self.is_outer = is_outer
        
        
    def run_all_steps(self):
        """ This will run all of the steps for the training process 
            Training itself depends on the state of the training fold.
            It checks for insufficient dataset.
        """
        # Load in the previously saved fold info. Check if valid. If so, use it.
        prev_info = self.load_state()
        # prev_info = None
        if prev_info is not None and \
        self.fold_info.testing_subject == prev_info.testing_subject and \
        self.fold_info.rotation_subject == prev_info.rotation_subject:
            self.fold_info = prev_info
            self.load_checkpoint()
            # Conditional if there is no checkpoint
            
            print(colored("Loaded previous existing state for testing subject " + 
                          f"{prev_info.testing_subject} and subject {prev_info.rotation_subject}.", 'cyan'))

        # Compute everything if no checkpoint
        else:
            self.fold_info.run_all_steps()
            self.save_state()
            
        # Create the datasets and train. (Datasets cannot be logged.)
        if self.create_dataset():
            self.train_model()
            self.output_results()
        
    
    def load_state(self):
        """ Loads the latest training state. """
        log = read_log_items(
            self.fold_info.config['output_path'],
            self.fold_info.config['job_name'], 
            ['fold_info']
        )
        if log is None or 'fold_info' not in log:
            return None
        return log['fold_info']
            
    
    def save_state(self):
        """ Saves the state of the fold to a log. """
        write_log(
            self.fold_info.config['output_path'], 
            self.fold_info.config['job_name'], 
            {'fold_info': self.fold_info},
            use_lock=self.fold_info.rank!=None
        )
    
    
    def load_checkpoint(self):
        """ Loads the latest checkpoint to start from. """
        results = get_most_recent_checkpoint(
            os.path.join(self.fold_info.config['output_path'], 'checkpoints'),
            self.fold_info.checkpoint_prefix
        )
        if results is not None:
            print(colored(f"Loaded most recent checkpoint of epoch: {results[1]}.", 'cyan'))
            self.fold_info.model.model = results[0]
            self.checkpoint_epoch = results[1]
        else:
            # Create model if not checkpoint
            self.fold_info.create_model()
        
        
    def create_dataset(self):
        """ Create the dataset needed for training the model.
            It will map the image paths into their respective image and label pairs.
            TODO Complete any image-reading changes here for different file types.
        """
        # Get the datasets for each phase
        for dataset in self.fold_info.datasets:
            if not self.fold_info.datasets[dataset]['files']:
                continue
            
            # Create the special image type readers
            csvreader = ImageReaderCSV(configs=self.fold_info.config)
            imreader = ImageReaderGlobal()
            
            
            b_drop_remainder = False 
            
            if dataset == "training":
                residual = len(self.fold_info.datasets[dataset]['files']) % self.fold_info.config['hyperparameters']['batch_size']
                print("Residual for Batch training  =", residual)
                
                if residual < (self.fold_info.config['hyperparameters']['batch_size']/2):
                    b_drop_remainder = True
                    print("Residual discarded")
                else:
                    print("Residual not discarded")
                
            
            # ds = tf.data.Dataset.from_tensor_slices(self.fold_info.datasets[dataset]['files'])
            
            
            # Parse images here
            ds = tf.data.Dataset.from_tensor_slices(self.fold_info.datasets[dataset]['files'])

            # Eager mode
            # ds_map = ds.map(lambda x: tf.py_function(
            #     func=parse_image,
            #     inp=[
            #         x,                                                                 # Filename
            #         self.fold_info.config['class_names'],                              # Class Names
            #         self.fold_info.config['hyperparameters']['channels'],              # Channels
            #         self.fold_info.config['hyperparameters']['do_cropping'],           # Do Cropping
            #         self.fold_info.config['hyperparameters']['cropping_position'][0],  # Offset Height
            #         self.fold_info.config['hyperparameters']['cropping_position'][1],  # Offset Width
            #         self.fold_info.config['target_height'],                            # Target Height
            #         self.fold_info.config['target_width'],                             # Target Width
            #         self.fold_info.label_position,                                     # Label Position
            #     ],
            #     Tout=[tf.float32, tf.int64]                
            # ))
            
            # Non eager version
            ds_map =ds.map(lambda x: parse_image(
                        x,                                                                 # Filename
                        self.fold_info.config['class_names'],                              # Class Names
                        self.fold_info.config['hyperparameters']['channels'],              # Channels
                        self.fold_info.config['hyperparameters']['do_cropping'],           # Do Cropping
                        self.fold_info.config['hyperparameters']['cropping_position'][0],  # Offset Height
                        self.fold_info.config['hyperparameters']['cropping_position'][1],  # Offset Width
                        self.fold_info.config['target_height'],                            # Target Height
                        self.fold_info.config['target_width'],                             # Target Width
                        self.fold_info.label_position
                        ),
                        num_parallel_calls=tf.data.AUTOTUNE,
                        deterministic=True )
            
             
                      
            self.fold_info.datasets[dataset]['ds'] = ds_map.batch(self.fold_info.config['hyperparameters']['batch_size'], drop_remainder=b_drop_remainder)
            
        # If the datasets are empty, cannot train
        if self.fold_info.datasets['training']['ds'] is None or \
           (not self.is_outer and self.fold_info.datasets['validation']['ds'] is None):
            print(colored(
                f"Non-fatal Error: training was skipped for the Test Subject {self.fold_info.testing_subject} and Subject {self.fold_info.rotation_subject}. " + 
                f"There were no files in the training and/or validation datasets.\n",
                'yellow'
            ))
            return False
        return True 
        
        
    def train_model(self):
        """ Train the model, assuming the given dataset is valid. """  
        if self.checkpoint_epoch != 0 and \
           self.checkpoint_epoch == self.fold_info.n_epochs:
            print(colored("Maximum number of epochs reached from checkpoint.", 'yellow'))
            return
        
        # Outer loop has no validation data
        if self.is_outer:
            validation_data = None
        else:
            validation_data = self.fold_info.datasets['validation']['ds']
        
        # Configuration for tensorboard

                
        if self.is_outer:
            file_prefix = f"{self.fold_info.model.model_type}_{self.fold_info.rotation_subject}_t_{self.fold_info.testing_subject}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        else:
            file_prefix = f"{self.fold_info.model.model_type}_{self.fold_info.fold_index}_t_{self.fold_info.testing_subject}_v_{self.fold_info.rotation_subject}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        path_output = Path(self.fold_info.config['output_path']).joinpath("tensorboard_output",file_prefix)
        # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        # with profile
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=path_output, histogram_freq=1,
        #                                                       profile_batch=(5,20))    

        # Just tensorboard
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=path_output, histogram_freq=1)    
        
        # Fit the model
        
        print("tf.executing_eagerly() =", tf.executing_eagerly())
            
        time_start = perf_counter()
        self.history = self.fold_info.model.model.fit(
            self.fold_info.datasets['training']['ds'],
            validation_data=validation_data,
            epochs=self.fold_info.n_epochs,
            initial_epoch=self.checkpoint_epoch,
            callbacks=[self.fold_info.callbacks]
        )
        self.time_elapsed = perf_counter() - time_start
        
        
        
    def output_results(self):
        """ Output the training results to file. """
        print(colored(f"Finished training for testing subject {self.fold_info.testing_subject} and subject {self.fold_info.rotation_subject}.", 'green'))
        output_results(
            os.path.join(self.fold_info.config['output_path'], 'training_results'), 
            self.fold_info.testing_subject, 
            self.fold_info.rotation_subject, 
            self.fold_info.fold_index, 
            self.fold_info.model, 
            self.history, 
            self.time_elapsed, 
            self.fold_info.datasets, 
            self.fold_info.config['class_names'],
            self.fold_info.config['job_name'],
            self.fold_info.config['selected_model_name'],
            self.is_outer,
            self.fold_info.rank
        )
        
