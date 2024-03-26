#!/bin/bash
# export CUBLAS_WORKSPACE_CONFIG=:4096:8
python src/misinformation_detection.py --source claim_article_pred --context none
python src/misinformation_detection.py --source claim_article_pred --context none
python src/misinformation_detection.py --source claim_article_pred --context none
python src/misinformation_detection.py --source claim_article_pred --context none

python src/misinformation_detection.py --source claim_article_pred --context low
python src/misinformation_detection.py --source claim_article_pred --context low
python src/misinformation_detection.py --source claim_article_pred --context low
python src/misinformation_detection.py --source claim_article_pred --context low

python src/misinformation_detection.py --source claim_article_pred --context high
python src/misinformation_detection.py --source claim_article_pred --context high
python src/misinformation_detection.py --source claim_article_pred --context high
python src/misinformation_detection.py --source claim_article_pred --context high
