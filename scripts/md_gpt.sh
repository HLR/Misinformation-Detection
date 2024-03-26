# !/bin/bash

python src/gpt_evaluator_new.py --source gt_strategy --model gpt-3.5-turbo
python src/gpt_evaluator_new.py --source pred_strategy --model gpt-3.5-turbo --context high

python src/gpt_evaluator_new.py --source claim --model gpt-3.5-turbo
python src/gpt_evaluator_new.py --source article --model gpt-3.5-turbo
python src/gpt_evaluator_new.py --source claim_article --model gpt-3.5-turbo

python src/gpt_evaluator_new.py --source claim_pred --model gpt-3.5-turbo --context high
python src/gpt_evaluator_new.py --source claim_gt --model gpt-3.5-turbo

python src/gpt_evaluator_new.py --source claim_article_gt --model gpt-3.5-turbo


python src/gpt_evaluator_new.py --source claim_article_pred --model gpt-3.5-turbo --context high

####################################################################################################

python src/gpt_evaluator_new.py --source gt_strategy --model gpt-4
python src/gpt_evaluator_new.py --source pred_strategy --model gpt-4 --context high

python src/gpt_evaluator_new.py --source claim --model gpt-4
python src/gpt_evaluator_new.py --source article --model gpt-4
python src/gpt_evaluator_new.py --source claim_article --model gpt-4

python src/gpt_evaluator_new.py --source claim_pred --model gpt-4 --context high
python src/gpt_evaluator_new.py --source claim_gt --model gpt-4
python src/gpt_evaluator_new.py --source claim_article_gt --model gpt-4

python src/gpt_evaluator_new.py --source claim_article_pred --model gpt-4 --context high

