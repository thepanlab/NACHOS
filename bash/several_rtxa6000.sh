cd /home/pcallec/mif_outer/scripts

mpirun -n 4 --hosts 10.254.214.32:3,10.254.214.37:1 python3 -m training.training_multiprocessing.loop_inner.multiprocessed_training_inner_loop --file /home/pcallec/mif_outer/scripts/training/training_config_files/loop_inner/distribution/rtxa6000_rs_0_config_ngpu3.json --ngpus 2 | tee distributed_rtxa600_rs0_ngpu3.txt

mpirun -n 5 --hosts 10.254.214.32:3,10.254.214.37:2 python3 -m training.training_multiprocessing.loop_inner.multiprocessed_training_inner_loop --file /home/pcallec/mif_outer/scripts/training/training_config_files/loop_inner/distribution/rtxa6000_rs_0_config_ngpu4.json --ngpus 2 | tee distributed_rtxa600_rs0_ngpu4.txt

mpirun -n 9 --hosts 10.254.214.32:3,10.254.214.37:2,10.254.214.39:2,10.254.214.46:2 python3 -m training.training_multiprocessing.loop_inner.multiprocessed_training_inner_loop --file /home/pcallec/mif_outer/scripts/training/training_config_files/loop_inner/distribution/rtxa6000_rtx4090_rs_0_config_ngpu8.json --ngpus 2 | tee distributed_rtxa600_rtx4090_rs0_ngpu8.txt
