#!/bin/bash
###### Skip x2 supervision ##### 
module load anaconda/2022.10
module load cuda/11.8
source activate torch
export PYTHONUNBUFFERED=1

start_time=$(date +%s)
MODEL_NAME1="TimeLens"
PARAM_NAME1="timelens_RC_x8_lpips_datav2_selftuning_x2" 
MODEL_PRETRAINED1="./pretrained_weights/init.pt"
FLAGS1="--save_flow True"
# FLAGS2="--skip_training "
for idx in {0..26} 
do
    echo "$idx"
    FLAGS3="--STN $idx"
    python run_network.py --model_name "$MODEL_NAME1" --param_name "$PARAM_NAME1" --model_pretrained "$MODEL_PRETRAINED1" $FLAGS1 $FLAGS2 $FLAGS3
done
end_time=$(date +%s)
run_time=$((end_time - start_time))
echo "Iteration $idx completed in $run_time seconds."


# ##### Random skip ##### 
# start_time=$(date +%s)

# MODEL_NAME1="TimeLens"
# PARAM_NAME1="timelens_RC_x8_lpips_datav2_selftuning_mix" 
# MODEL_PRETRAINED1="/ailab/user/zhangziran/code/Sim2Real_release/weight/TimeLens_12.pt"
# FLAGS1="--save_flow True"
# # FLAGS2="--skip_training "
# for idx in {0..1}
# do
#     echo "$idx"
#     FLAGS3="--STN $idx"
#     python run_network.py --model_name "$MODEL_NAME1" --param_name "$PARAM_NAME1" --model_pretrained "$MODEL_PRETRAINED1" $FLAGS1 $FLAGS2 $FLAGS3
# done
# end_time=$(date +%s)
# run_time=$((end_time - start_time))
# echo "Iteration $idx completed in $run_time seconds."