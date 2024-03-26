#!/bin/bash
# export CUBLAS_WORKSPACE_CONFIG=:4096:8
python src/misinformation_detection_test.py --source claim
python src/misinformation_detection_test.py --source article
python src/misinformation_detection_test.py --source claim_article

python src/misinformation_detection_test.py --source gt_strategy
python src/misinformation_detection_test.py --source pred_strategy --context none
python src/misinformation_detection_test.py --source pred_strategy --context low
python src/misinformation_detection_test.py --source pred_strategy --context high

python src/misinformation_detection_test.py --source claim_gt
python src/misinformation_detection_test.py --source claim_pred --context none
python src/misinformation_detection_test.py --source claim_pred --context low
python src/misinformation_detection_test.py --source claim_pred --context high

python src/misinformation_detection_test.py --source claim_article_gt
python src/misinformation_detection_test.py --source claim_article_pred --context none
python src/misinformation_detection_test.py --source claim_article_pred --context low
python src/misinformation_detection_test.py --source claim_article_pred --context high

