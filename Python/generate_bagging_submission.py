# -*- coding: UTF-8 -*-

"""
Generate the final submission.
If everything goes well, this is supposed to give:
public LB: 0.98591
private LB: 0.98527

Python version: 2.7.6
Version: 1.0 at Sep 01 2014
Author: Chenglong Chen < yr@Kaggle >
Email: c.chenglong@gmail.com
"""

import os
import re
import natsort
import numpy as np
import pandas as pd

def getFileNames(folder, end):
    
    file = [ folder+f for f in os.listdir(folder) if len(re.findall("^[0-9]+.csv",f))>0 ]
    file = natsort.natsorted(file, key=lambda s: s.lower())
    
    return file
        
def getRankMatrix(ensFiles, names):
    
    for i,f in enumerate(ensFiles):
        print "Read in #%s: %s" % (i+1,f)
        df = pd.read_csv(f)
        df[names[i]] = np.arange(df.shape[0])
        if i == 0:
            rankMatrix = df.copy()
        else:
            rankMatrix = pd.merge(left=rankMatrix, right=df, on="id")
            
    return rankMatrix
    
def main():
    model = ["mixed", "proved"]
    LOSS = ["logistic", "logistic"]
    LR = [0.5, 0.2]
    LRDECAY = [1.0, 1.0]
    PASS = [10, 16]
    all_config = []
    for l in xrange(len(LOSS)):
        this_config = "%s_%s_lr%s_lrdecay%s_pass%s" % (model[l], LOSS[l], LR[l], LRDECAY[l], PASS[l])
        all_config.append( "["+this_config+"]" )
        ensFilesFolder = "../Submission/%s/" % this_config
        ensFiles = getFileNames(ensFilesFolder, ".csv")[:50]
        print "Found %s files to ensemble for: model=%s, loss=%s, lr=%s, pass=%s" % (len(ensFiles), model[l], LOSS[l], LR[l], PASS[l])
        medianEnsSubmission = ensFilesFolder + "%s_ensemble%s_median.csv" % (this_config, len(ensFiles))
        meanEnsSubmission = ensFilesFolder + "%s_ensemble%s_mean.csv" % (this_config, len(ensFiles))
    
        rankPredictors = [ "%s_rank.%s"%(this_config, i+1) for i in xrange(len(ensFiles))]
        rankMatrix = getRankMatrix(ensFiles, rankPredictors)
        if l == 0:
            rankMatrixAll = rankMatrix.copy()
            rankPredictorsAll = rankPredictors
        else:
            rankMatrixAll = pd.merge(left=rankMatrixAll, right=rankMatrix, on="id")
            rankPredictorsAll += rankPredictors
            
        rankMatrix["median_score"] = rankMatrix[rankPredictors].median(axis=1)
        rankMatrix["mean_score"] = rankMatrix[rankPredictors].mean(axis=1)
        
        # sort the result by ensemble score and break ties by the other ensemble score
        # using median method    
        item_id = rankMatrix.sort(["median_score","mean_score"], ascending=True)["id"]
        dfSub = pd.DataFrame({"id": item_id})
        dfSub.to_csv(medianEnsSubmission, index=False)
        # using mean method
        item_id = rankMatrix.sort(["mean_score","median_score"], ascending=True)["id"]
        dfSub = pd.DataFrame({"id": item_id})
        dfSub.to_csv(meanEnsSubmission, index=False)
      
    ##
    # final ensemble
    Weights = [1]*50+[0]*50
    rankMatrixAll["weighted_average_score"] = rankMatrixAll[rankPredictorsAll].dot(Weights)
    rankMatrixAll["mean_score"] = rankMatrixAll[rankPredictorsAll].mean(axis=1)
        
    # sort the result by ensemble score and break ties by the other ensemble score
    # using weighted average method
    item_id = rankMatrixAll.sort(["mean_score","weighted_average_score"], ascending=True)["id"]
    dfSub = pd.DataFrame({"id": item_id})
    fileName = "../Submission/%s_ensemble_weighted_average.csv"%"_".join(all_config)
    dfSub.to_csv(fileName, index=False)
    
if __name__ == "__main__":
    main()
