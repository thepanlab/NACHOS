cd /home/pcallec/mif_outer/scripts

mpirun -n 6 --hosts 10.254.214.32:3,10.254.214.46:3 python3 -m training.training_multiprocessing.loop_inner.multiprocessed_training_inner_loop --file /home/pcallec/mif_outer/scripts/training/training_config_files/loop_inner/distribution/temp_rs_0_config_ngpu4.json --ngpus 2 --dummy| tee distributed_temp_rs0_ngpu4.txt

mpirun -n 3 --hosts 10.254.214.32:3 python3 -m training.training_multiprocessing.loop_inner.multiprocessed_training_inner_loop --file /home/pcallec/mif_outer/scripts/training/training_config_files/loop_inner/distribution/temp_rs_0_config_ngpu4.json --ngpus 2 --dummy| tee distributed_temp_rs0_ngpu4.txt