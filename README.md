# Kaggle's The Hunt for Prohibited Content Competition
  
This repo holds the code I used to make submision to [Kaggle's The Hunt for Prohibited Content Competition](http://www.kaggle.com/c/avito-prohibited-content). The score using this implementation is **0.98527**, ranking 4th out of 289 teams. (That entry is placed in `./Submission` folder.)

I initally entered this competition to familiarize myself with VW and Linux & Shell (I used to be a Windows user). So the code provided here might not be as efficient and elegant as they can be.


## Method

* It uses LR to build classifier on a bunch of features including

 - BOW/Tf-idf 1/2gram features of the title, description, attributes, etc.
 - All the raw features such as category, subcategory, price, etc.
 - Some cross-features between the above features, such as subcategory & price, etc seem to help a lot.

* I initally trained on the whole dataset, and later found some imporvement by ensembling ranking predicitions from a model using only is_proven bloced ads and unblockded ads.

* I have tried all the cost functions provided in VW, i.e., log-loss, hinge loss, squared loss, and quantile loss, but found log-loss give consistently better results. Ensemble models from different loss doesn't seem to buy me anything.

## Code layout
* **Main functions**
 - `run_all.sh` : run everything in one shot
 - `grid_search.sh` : perform grid search and bagging (called by `run_all.sh`)
 - `generate_vw_file.py`: generate VW format training and testing data (called by `run_all.sh`)
 - `generate_bagging_submission.py`: generate final bagging submission (called by `run_all.sh`)
 
* **Helper functions**
 - `generate_submission.py`: convert VW format prediction to Kaggle submission
 - `generate_weighted_sample.py`: convert training data to importance weighted one (used in grid search for the best sample weights)
 - `generate_bootstrap.py`: generate bootstrap samples (used in bagging)
 - `APatK.py`: compute AP@k (used in grid search)
 - `ngram.py`: construct n-gram
 

## Requirement

- [Vowpal Wabbit](https://github.com/JohnLangford/vowpal_wabbit): I used the latest version of VW for all the traininng.
- [gensim](http://radimrehurek.com/gensim/): I used gensim for extracting tf-idf features.
  
  
## Instruction

* download data from the [competition website](http://www.kaggle.com/c/avito-prohibited-content/data) and put all the data into `./Data` dir
* put all the code into `./Python` dir:
* run `bash ./Python/run_all.sh` to create csv submission to Kaggle.


## Discussion

* It seems promissing to train seperate model for each category as discussed [here](http://www.kaggle.com/c/avito-prohibited-content/forums/t/10178/congrats-barisumog-giulio/52856#post52856).
* Semi-supervised learning (SSL) is shown to be useful for the [winning team](http://www.kaggle.com/c/avito-prohibited-content/forums/t/10178/congrats-barisumog-giulio/52812#post52812). The idea of SSL is also exploited in another competition: [Kaggles' Greek Media Monitoring Multilabel Classification (WISE 2014)](http://www.kaggle.com/c/wise-2014) as shown [here](http://www.kaggle.com/c/wise-2014/forums/t/9773/our-approach-5th-place/50766#post50766).