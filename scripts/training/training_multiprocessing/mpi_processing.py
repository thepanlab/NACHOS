from training.training_modules.data_processing import training_preparation, fold_generator
from training.training_modules.output_processing import console_printing
from training.training_modules.training_processing import training_loop
from training.training_checkpointing_logging.logger import *
from util.get_config import parse_training_configs
from termcolor import colored
from datetime import date
import tensorflow as tf
from mpi4py import MPI
import datetime
import time
import math
import os
import sys

# Location of the configurations
CONFIG_LOC = './training/training_config_files'

def split_tasks(configs, n_proc, is_outer):
    """ Generates config-fold tuples for training.

    Args:
        configs (list of dict): List of configurations.
        n_proc (int): Number of training processes.
        is_outer (bool): If this is of the outer loop or not.
        
    Returns:
        (list tuples): A list of config-fold tuples.
    """
    # Create a list of (config, test subject, validation subject) tuples
    tasks = []
    for config in configs:
        
        # Generate all fold-pairs
        test_subjects = config['test_subjects']
        validation_subjects = None if is_outer else config['validation_subjects']
        folds = fold_generator.generate_pairs(test_subject_list=test_subjects,
                                              validation_subject_list=validation_subjects,
                                              subject_list=config['subject_list'],
                                              do_shuffle=config['shuffle_the_folds'],
                                              param_epoch=config["hyperparameters"]['epochs'],
                                              is_outer=is_outer)
        print(folds)
        # Add folds to task list
        tasks.extend([(config, n_epochs, test_subject, validation_subject) for n_epochs, test_subject, validation_subject in folds])
    return tasks
        
            
def run_training(rank, config, n_epochs, test_subject, validation_subject, is_outer):
    """ Run the training loop for some task.
    Args:
        rank (int): The rank of the process.
        config (dict): The given training configuration for the task.
        test_subject (str): The task's testing subject name.
        validation_subject (str): The task's training/validation subject name. May be None.
        is_outer (bool): If this task is of the outer loop.
    """    
    # Read in the log, if it exists
    job_name = f"{config['job_name']}_test_{test_subject}" if is_outer else f"{config['job_name']}_test_{test_subject}_sub_{validation_subject}"
    log = read_log_items(
        config['output_path'], 
        job_name, 
        ['is_finished']
    )
    
    # If this fold has finished training, return
    if log and log['is_finished']:
        return
    
    # Run the subject pair
    if is_outer:
        print(colored(f"Rank {rank} is starting training for {test_subject}.", 'green'))
    else:
        print(colored(f"Rank {rank} is starting training for {test_subject} and validation subject {validation_subject}.", 'green'))
        
    subject_loop(rank, config, is_outer, n_epochs, test_subject, validation_subject=validation_subject)
    write_log(
        config['output_path'], 
        job_name, 
        {'is_finished': True},
        use_lock=True
    )
        
        
def subject_loop(rank, config, is_outer, n_epochs, test_subject, validation_subject=None):
    """ Executes the training loop for the given test subject.

    Args:
        rank (int): The rank of the process.
        config (dict): The training configuration.
        is_outer (bool): If this task is of the outer loop.
        test_subject (str): The task's test subject name.
        validation_subject (str): The task's training/validation subject name. May be None. (Optional)
    """
    print(colored(
        f"\n\n===========================================================\n" + 
        f"Rank {rank} is starting training for {test_subject} in {config['selected_model_name']}\n"
        , 'magenta'
    ))
    training_vars = training_preparation.TrainingVars(config, is_outer, test_subject, validation_subject=validation_subject)
    # training_loop(config, testing_subject, files, folds, rotations, indexes, label_position, n_epochs, is_outer, rank=None)
    training_loop.training_loop(
        config=config, 
        testing_subject=test_subject, 
        files=training_vars.files, 
        folds=training_vars.folds, 
        rotations=training_vars.n_folds, 
        indexes=training_vars.indexes, 
        label_position=training_vars.label_position,
        n_epochs=n_epochs,
        is_outer=is_outer,
        rank=rank
    )


def main(config_loc, is_outer):
    """ Runs the training process for each configuration and test subject. Process 0 DOES NOT train. 
    Args:
        config_loc (str): The location of the configuration.
        is_outer (bool): Whether this is of the outer loop. 
    """
    
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_proc = comm.Get_size()
    
    print("python location", os.path.dirname(sys.executable))
    
    # Initalize TF, set the visible GPU to rank%2 for rank > 0
    # tf_config = tf.compat.v1.ConfigProto()
    if rank == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        # tf_config.gpu_options.visible_device_list = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str((rank+1)%2)
        # tf_config.gpu_options.allow_growth = True
        
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    # session = tf.compat.v1.Session(config=tf_config)

    
    # Rank 0 initializes the program and runs the configuration loops
    if rank == 0:  
         
        # Get start time
        start_time_name = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        start_time = time.time()
        
        # Get the configurations
        configs = parse_training_configs(config_loc)
        n_configs = len(configs)
        next_task_index = 0
        
        # No configs, no run
        if n_configs == 0:
            print(colored("No configurations given.", 'yellow'))
            for subrank in range(1, n_proc):
                comm.send(False, dest=subrank)
            exit(-1)
        
        # Get the tasks for each process
        tasks = split_tasks(configs, n_proc, is_outer)
        print("tasks = ", tasks)
        # Listen for process messages while running
        exited = []
        while True:
            subrank = comm.recv(
                source=MPI.ANY_SOURCE
            )
            
            # Send task if the process is ready
            if tasks:
                print(colored(f"Rank 0 is sending rank {subrank} their first task.", 'green'))
                comm.send(
                    tasks.pop(), 
                    dest=subrank
                )
                next_task_index += 1
                    
            # If no task remains, terminate process
            else:
                print(colored(f"Rank 0 is terminating rank {subrank}, no tasks to give.", 'red'))
                comm.send(
                    False, 
                    dest=subrank
                )
                exited += [subrank]
                
                # Check if any processes are left, end this process if so
                if all(subrank in exited for subrank in range(1, n_proc)):
                    
                    # Get end time and print
                    print(colored(f"Rank 0 is printing the processing time.", 'red'))
                    elapsed_time = time.time() - start_time
                    if not os.path.exists("../results/training_timings"):
                        os.makedirs("../results/training_timings")
                    outfile = f'_TIME_MPI_OUTER_{start_time_name}.txt' if is_outer else f'_TIME_MPI_INNER_{start_time_name}.txt'
                    with open(os.path.join("../results/training_timings", outfile), 'w') as fp:
                        fp.write(f"{elapsed_time}")
                    print(colored(f'Rank {rank} terminated. All other processes are finished.', 'yellow'))
                    break
            
    # The other ranks will listen for rank 0's messages and run the training loop
    else: 
        # tf.config.run_functions_eagerly(True)
        
        # Listen for the first task
        print(colored(f'Rank {rank} is listening for process 0.', 'cyan'))
        comm.send(rank, dest=0)
        task = comm.recv(source=0)
        
        # While there are tasks to run, train
        while task:
                    
            # Training loop
            config, n_epochs, test_subject, validation_subject = task
            print(colored(f"rank {rank}: test {test_subject}, train {validation_subject}", 'cyan'))         
            
            run_training(rank, config, n_epochs, test_subject, validation_subject, is_outer)
            comm.send(rank, dest=0)
            task = comm.recv(source=0)
            
        # Nothing more to run.
        print(colored(f'Rank {rank} terminated. All jobs finished for this process.', 'yellow'))
        