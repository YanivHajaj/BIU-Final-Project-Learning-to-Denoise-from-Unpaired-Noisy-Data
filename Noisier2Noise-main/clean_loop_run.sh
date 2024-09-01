#!/usr/local/bin/bash

# Set the folder path you want to clean
folder_path="./csvs"

# Parse command line arguments
range_start=6
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

    # Check if the folder exists
    if [ -d "$folder_path" ]; then
        # Delete everything inside the folder
        rm -rf "$folder_path"/*
        echo "Deleted all files and folders inside: $folder_path"
    else
        echo "Folder not found: $folder_path"
    fi
fi


# python3 utils.py

# Skip the loop if skip_run_flag is set
if $skip_run_flag; then
    echo "Skipping the loop execution as --skip_run flag is set."
else

    # Run the loop
    for i in $(seq $range_start $range_end)
    do
        python3 test.py --exp_num 6 --n_epochs 200 --gpu_num 0 --dataset Set20 --exp_rep exp13 --test_info "testing median - quantile" --aver_num $i
    done
fi

# Check for the 'graph' parameter and perform the action if it's set
if $graph_flag; then
    echo "Graph generation process initiated."
    python3 graph.py --set Set20 --exp 13
fi

# usage example: 
# ./clean_loop_run.sh --range 30 --clean --graph --skip_run

# ./clean_loop_run.sh --graph --skip_run

# ./clean_loop_run.sh --range  

# ./clean_loop_run.sh --skip_run --graph

# ./clean_loop_run.sh --range 30 --clean --graph 

# python3 graph.py --set Set20 --exp 5