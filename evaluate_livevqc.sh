#!/bin/bash

python refine_yaml.py 10
python evaluate_one_dataset.py -t val-livevqc --output livevqc.csv -v 1