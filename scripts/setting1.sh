#!/bin/bash

CONFIG_PATH="./configs/setting1.yaml"

python train.py --config-path $CONFIG_PATH
python test.py --config-path $CONFIG_PATH