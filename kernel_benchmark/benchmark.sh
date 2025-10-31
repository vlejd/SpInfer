# source run_all_main.sh  # SpInfer does not support batch 1
gpu_name=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | head -n1 | sed -E 's/.*: (.*)/\1/' | tr ' ' '_')

mkdir -p cusparse_results_16bit/${gpu_name}/
mkdir -p sparta_results_16bit/${gpu_name}/
mkdir -p sputnik_results_16bit/${gpu_name}/

source run_cusparse.sh | tee cusparse_results_16bit/${gpu_name}/raw_results.txt
source run_sparta_4090.sh | tee sparta_results_16bit/${gpu_name}/raw_results.txt
source run_sputnik_4090.sh | tee sputnik_results_16bit/${gpu_name}/raw_results.txt