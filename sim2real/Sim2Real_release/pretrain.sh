# Define model parameters
MODEL_NAME1="TimeLens"
PARAM_NAME1="GOPRO_release_TimeLens_tuning"
MODEL_PRETRAINED1="./pretrained_weights/init.pt"
FLAGS1="--save_flow True"
# FLAGS2="--skip_training "
python run_network.py --model_name "$MODEL_NAME1" --param_name "$PARAM_NAME1" --model_pretrained "$MODEL_PRETRAINED1" $FLAGS1 $FLAGS2 