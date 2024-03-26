#!/bin/bash
# export CUBLAS_WORKSPACE_CONFIG=:4096:8
python src/misinformation_detection.py --source pred_strategy --context none
python src/misinformation_detection.py --source pred_strategy --context none
python src/misinformation_detection.py --source pred_strategy --context none
python src/misinformation_detection.py --source pred_strategy --context none

python src/misinformation_detection.py --source pred_strategy --context low
python src/misinformation_detection.py --source pred_strategy --context low
python src/misinformation_detection.py --source pred_strategy --context low
python src/misinformation_detection.py --source pred_strategy --context low

python src/misinformation_detection.py --source pred_strategy --context high
python src/misinformation_detection.py --source pred_strategy --context high
python src/misinformation_detection.py --source pred_strategy --context high
python src/misinformation_detection.py --source pred_strategy --context high