#!/bin/bash

PYTHONPATH="$(dirname $0)/../../":$PYTHONPATH

python ./tools/detection/inference.py -input ./data/xyb/images/val -output ./results_show
