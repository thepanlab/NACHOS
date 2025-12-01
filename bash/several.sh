cd /home/pcallec/mif_outer/scripts

mpirun -n 9 --hosts 10.254.214.32:3,10.254.214.27:2,10.254.214.39:2,10.254.214.46:2 python3 -m training.training_multiprocessing.loop_inner.multiprocessed_training_inner_loop --file /home/pcallec/mif_outer/scripts/training/training_config_files/loop_inner/distribution/rs_0_config_ngpu8.json --ngpus 2  | tee distributed_rs0_ngpu8.txt

mpirun -n 3 --hosts 10.254.214.32:3 python3 -m training.training_multiprocessing.loop_inner.multiprocessed_training_inner_loop --file /home/pcallec/mif_outer/scripts/training/training_config_files/loop_inner/distribution/rs_0_config_ngpu2.json --ngpus 2 | tee distributed_rs0_ngpu2.txt

mpirun -n 2 --hosts 10.254.214.32:2 python3 -m training.training_multiprocessing.loop_inner.multiprocessed_training_inner_loop --file /home/pcallec/mif_outer/scripts/training/training_config_files/loop_inner/distribution/rs_0_config_ngpu1.json --ngpus 1 | tee distributed_rs0_ngpu1.txt