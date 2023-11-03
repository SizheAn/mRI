#! /bin/bash

# Define the possible values for each parameter
protocols=(1 2)
datasplits=(1 2)
modalities=("radar" "imu")

# Loop over all combinations of parameter values
for protocol in ${protocols[@]}; do
  for datasplit in ${datasplits[@]}; do
    for modality in ${modalities[@]}; do
      # Define the log file path
      log_file="model/${modality}/log/protocol${protocol}_datasplit${datasplit}_train.log"
      # Create the necessary directories
      mkdir -p $(dirname $log_file)
      # Run the Python script and redirect output to the log file
      python multimodal_train.py -p $protocol -s $datasplit -m $modality 2>&1 | tee $log_file
    done
  done
done