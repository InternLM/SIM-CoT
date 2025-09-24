#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# Create data directory if it doesn't exist
mkdir -p data
# Download and process GSM8K dataset for Internalize CoT

wget https://media.githubusercontent.com/media/da03/Internalize_CoT_Step_by_Step/e06a32ee5e4cd117171daeb4755d2a97ece62761/data/gsm8k/train.txt -O data/gsm_train.txt
wget https://raw.githubusercontent.com/da03/Internalize_CoT_Step_by_Step/e06a32ee5e4cd117171daeb4755d2a97ece62761/data/gsm8k/valid.txt -O data/gsm_valid.txt
wget https://raw.githubusercontent.com/da03/Internalize_CoT_Step_by_Step/e06a32ee5e4cd117171daeb4755d2a97ece62761/data/gsm8k/test.txt -O data/gsm_test.txt

for split in train valid test; do
  python preprocessing/gsm_icot.py ${split}
  rm data/gsm_${split}.txt
done