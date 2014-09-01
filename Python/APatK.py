# -*- coding: UTF-8 -*-

"""
Python version: 2.7.6
Version: 1.0 at Sep 01 2014
Author: Chenglong Chen < yr@Kaggle >
Email: c.chenglong@gmail.com
"""

import argparse
import numpy as np
import pandas as pd
import cPickle as pkl
from csv import DictReader
from datetime import datetime


def getRelevantIDs(tsvFile, verbose=True):
    """ Get all relevant IDs
    You only need to run this once
    """
    
    start = datetime.now()
    relevantIDs = []
    with open(tsvFile, "rb") as tsvReader:
        itemReader = DictReader(tsvReader, delimiter='\t', quotechar='"')
        for i, item in enumerate(itemReader):
            item = {featureName:featureValue.decode('utf-8') \
                    for featureName,featureValue in item.iteritems() \
                    if featureValue is not None}
            
            if item["is_blocked"] == "1":
                relevantIDs.append( int(item["itemid"]) ) 

            if (i+1)%10000 == 0 and verbose:
                print( "%s\t%s"%((i+1),str(datetime.now() - start)) )

    return relevantIDs
    
    
def computeAPatK ( dfSub ):
    """Calculates AP@k given a file with IDs
    sorted in order of relevance
    """
    
    # load all relevant IDs stored in pkl format
    dataPath = "../Data/"
    try:
        with open(dataPath + "relevantIDs.pkl", "rb") as f:
            relevantIDs = pkl.load(f)
    except:
        tsvFileTrain = dataPath + "avito_train.tsv"
        relevantIDs = getRelevantIDs(tsvFileTrain, verbose=False)
        with open(dataPath + "relevantIDs.pkl", "wb") as f:
            pkl.dump(relevantIDs, f, -1)
    
    # read in submission file
    if type(dfSub) == str:
        dfSub = pd.read_csv(dfSub)
    numSample = dfSub.shape[0]
    # get the value of K
    ratio = 32500/(1351243*0.5)
    K = int(np.floor(numSample*ratio))
    
    # compute AP@k
    relevant = np.asarray(dfSub[:K].isin(relevantIDs), dtype=float)
    APatK_ = np.mean( np.cumsum(relevant)/np.arange(1,K+1) )
    
    return APatK_


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="compute AP@k")
    parser.add_argument("input")
    args = parser.parse_args()
    #print "--------------------"
    APatK_ = computeAPatK( args.input )
    print "%s" % np.round(APatK_, 8)
    #print "--------------------"