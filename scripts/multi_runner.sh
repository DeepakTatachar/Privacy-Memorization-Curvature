#!/bin/bash

# Get the list of host-GPU pairs and job files from the user
host_gpu_file=$1
jobs_file=$2
environment_name='py3.11_tf'
wd='set the root directory to start runs from'

# Check if the host-GPU file and jobs file were provided
if [ -z "$host_gpu_file" ] || [ -z "$jobs_file" ]; then
  echo "Please provide the host-GPU file and jobs file."
  exit 1
fi

# Initialize empty lists to store hosts and GPU IDs
hosts=()
gpu_ids=()

# Read the host-GPU pairs from the file
while IFS=, read -r host gpu_id; do
    # Add the host and GPU ID to the corresponding lists
    hosts+=($host)
    gpu_ids+=($gpu_id)
done < $host_gpu_file

# Get the total number of lines in the jobs file
num_jobs=$(wc -l $jobs_file | cut -d' ' -f1)
# Convert the number of lines to an integer
num_jobs=$((num_jobs))

# Initialize a variable to track the number of running jobs
running_jobs=0

# Iterate over the list of jobs
job_id=0

mapfile -t jobs < $jobs_file
while [ $job_id -lt $num_jobs ]; do
    # Check if there are available hosts and GPUs
    available_hosts=("${hosts[@]}")
    available_gpu_ids=("${gpu_ids[@]}")

    # Get the total number of lines in the jobs file
    num_jobs=$(wc -l $jobs_file | cut -d' ' -f1)
    # Convert the number of lines to an integer
    num_jobs=$((num_jobs))

    # Process jobs until there are no available hosts or GPUs
    while [ "${#available_hosts[@]}" -gt 0 ] && [ "${#available_gpu_ids[@]}" -gt 0 ]; do
        # Get the next job from the queue
        current_job=${job_queue[0]}

        # Remove the job from the queue
        unset job_queue[0]
        job_queue=("${job_queue[@]}")

        # Get the next available host
        current_host=${available_hosts[0]}

        # Remove the host from the available hosts list
        unset available_hosts[0]
        available_hosts=("${available_hosts[@]}")

        # Get the next available GPU ID
        current_gpu_id=${available_gpu_ids[0]}

        # Remove the GPU ID from the available GPU IDs list
        unset available_gpu_ids[0]
        available_gpu_ids=("${available_gpu_ids[@]}")

        current_job=${jobs[$job_id]}
        full_job="CUDA_VISIBLE_DEVICES=$current_gpu_id $current_job --gpu_id=$current_gpu_id"
        echo "$full_job"
        # Start the job on the current host, setting the CUDA_VISIBLE_DEVICES environment variable to the current GPU ID
        ssh $current_host ". ~/.bashrc ; conda activate $environment_name; cd $wd;$full_job"  &

        # Increment the number of running jobs
        running_jobs=$((running_jobs+1))
        job_id=$((job_id+1))

        # Print a message indicating that the job has been started
        echo "Started job '$full_job' on host '$current_host' using GPU $current_gpu_id"
    done

    # Wait for a running job to finish before starting the next job
    wait

    # Decrement the number of running jobs
    running_jobs=$((running_jobs-1))
done