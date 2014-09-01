# -*- coding: UTF-8 -*-

"""
Python version: 2.7.6
Version: 1.0 at Sep 01 2014
Author: Chenglong Chen < yr@Kaggle >
Email: c.chenglong@gmail.com
"""

import argparse
import pandas as pd

def vw_to_kaggle(raw_pred_file, sub_file):
    
    scores = []
    ids = []    
    with open(raw_pred_file) as f:
        for e,line in enumerate(f):
            row = line.strip().split(" ")
            scores.append( float(row[0]) )
            ids.append( int(row[1]) )

    scoreMatrix = pd.DataFrame({"id": ids, "score": scores})
    item_id = scoreMatrix.sort("score", ascending=False)["id"]
    dfSub = pd.DataFrame({"id": item_id})
    dfSub.to_csv(sub_file, index=False)
    
                
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="convert raw prediction to Kaggle submission")
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()    
    vw_to_kaggle(args.input, args.output)
