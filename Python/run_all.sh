#!/bin/bash
#####################################################################
# Code provided here for creating 4th place submission to           # 
# Kaggle's The Hunt for Prohibited Content Competition              #
# http://www.kaggle.com/c/avito-prohibited-content                  #
#                                                                   #
# Version: 1.0 at Sep 01 2014                                       #
# Author: Chenglong Chen < yr@Kaggle >                              #
# Email: c.chenglong@gmail.com                                      #
#####################################################################

# generate VW training and testing data
python generate_vw_file.py

# perform grid search and bagging for mixed & proved models
sudo bash grid_search.sh mixed
sudo bash grid_search.sh proved

# generate final submission
python generate_bagging_submission.py
