#!/bin/bash

# Automation that tests different values of alpha (0.2, 0.5, etc.)

# Define an array of alpha values and corresponding noise values
alpha_values=(1.0 0.8 0.5 0.3 0.1 0.00001)

# Iterate over the parameter sets
for i in "${!alpha_values[@]}"; do
  alpha=${alpha_values[$i]}
  
  # Run the Python script with the current parameters
  python3 test.py --exp_num 6 --n_epochs 200 --gpu_num 0 --test_info "testing with alpha ${alpha} and noise gauss_25" --dataset Set20 --alpha "${alpha}"
done



# noisier = noise
# python3 test.py --exp_num 6 --n_epochs 200 --gpu_num 0 --test_info "testing using noisier = noise, avernum = 20" --dataset Set22 --aver_num 20

# python3 test.py --exp_num 6 --n_epochs 200 --gpu_num 0 --test_info "testing using noisier = noise" --dataset Set22
# python3 test.py --exp_num 6 --n_epochs 200 --gpu_num 0 --test_info "testing trimmed" --dataset Set20


#   python3 test.py --exp_num 6 --n_epochs 200 --gpu_num 0 --test_info "testing with alpha ${alpha} and noise ${noise}" --dataset Set22 --alpha "${alpha}" --noise "${noise}"


# ./run_tests.sh