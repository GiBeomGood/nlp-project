#!/bin/bash

CONFIG_PATH="./configs/setting6.yaml"

python train.py --config-path $CONFIG_PATH
python test.py --config-path $CONFIG_PATH