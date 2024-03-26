Real-World Multi-Domain Dataset for Evidence-Based Fact Checking of Claims

The MultiFC is the largest publicly available dataset of naturally occurring factual claims for automatic claim verification.
It is collected from 26 English fact-checking websites paired with textual sources and rich metadata and labeled for veracity by human expert journalists.


###### TRAIN and DEV #######
The train and dev files are (tab separated) and contain the following metadata:
claimID, claim, label, claimURL, reason, categories, speaker, checker, tags, articleTitle, publishDate, claimDate, entities

Fields that could not be crawled were set as "None." Please refer to Table 11 of our paper to see the summary statistics.


###### TEST #######
The test file follows the same structure. However, we have removed the label. Thus, it only presents 12 metadata.
claimID, claim, claimURL, reason, categories, speaker, checker, tags, articleTitle, publishDate, claimDate, entities

Fields that could not be crawled were set as "None." Please refer to Table 11 of our paper to see the summary statistics.


###### Snippets ######
The text of each claim is submitted verbatim as a query to the Google Search API (without quotes).
In the folder snippet, we provide the top 10 snippets retrieved. In some cases you fewer snippets are provided
since we have excluded the claimURL from the snippets.
Each file in the snippets folder is named after the claimID of the claim submitted as a query.
Snippets file is (tab-separated) and contains the following metadata:
rank_position, title, snippet, snippet_url


For more information, please refer to our paper:
References:
Isabelle Augenstein, Christina Lioma, Dongsheng Wang, Lucas Chaves Lima, Casper Hansen, Christian Hansen, and Jakob Grue Simonsen. 2019. 
MultiFC: A Real-World Multi-Domain Dataset for Evidence-Based Fact Checking of Claims. In EMNLP. Association for Computational Linguistics.

https://copenlu.github.io/publication/2019_emnlp_augenstein/
