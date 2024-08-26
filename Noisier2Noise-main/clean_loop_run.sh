#!/usr/local/bin/bash

# Hardcoded files to delete
file_to_delete1="SSIM_0.csv"
file_to_delete2="SSIM_1.csv"
file_to_delete3="SSIM_2.csv"
file_to_delete4="SSIM_all_images_average.csv"
file_to_delete5="PSNR_0.csv"
file_to_delete6="PSNR_1.csv"
file_to_delete7="PSNR_2.csv"
file_to_delete8="PSNR_all_images_average.csv"

# Parse command line arguments
range_start=1
range_end=20
clean_flag=false
graph_flag=false
skip_run_flag=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --range) 
            IFS='-' read -r range_start range_end <<< "$2"
            shift 2
            ;;
        --clean)
            clean_flag=true
            shift
            ;;
        --graph)
            graph_flag=true
            shift
            ;;
        --skip_run)
            skip_run_flag=true
            shift
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
done

# Execute clean operation if flag is set
if $clean_flag; then
    for file_to_delete in "$file_to_delete1" "$file_to_delete2" "$file_to_delete3" "$file_to_delete4" "$file_to_delete5" "$file_to_delete6" "$file_to_delete7" "$file_to_delete8"
    do
        if [ -f "$file_to_delete" ]; then
            rm "$file_to_delete"
            echo "Deleted file: $file_to_delete"
        else
            echo "File not found: $file_to_delete"
        fi
    done
fi

python3 utils.py

# Skip the loop if skip_run_flag is set
if $skip_run_flag; then
    echo "Skipping the loop execution as --skip_run flag is set."
else
    # Run the loop
    for i in $(seq $range_start $range_end)
    do
        python3 test.py --exp_num 6 --n_epochs 200 --gpu_num 0 --dataset Set20 --aver_num $i
    done
fi

# Check for the 'graph' parameter and perform the action if it's set
if $graph_flag; then
    echo "Graph generation process initiated."
    python3 graph.py
fi

# usage example: 
# ./clean_loop_run.sh --range 30 --clean --graph --skip_run

# ./clean_loop_run.sh --graph --skip_run

# ./clean_loop_run.sh --range 1  

# ./clean_loop_run.sh --skip_run --clean
