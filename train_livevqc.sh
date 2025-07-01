#!/bin/bash

python refine_yaml.py 10
python train_one_dataset.py -t val-livevqc
