# -*- coding: UTF-8 -*-

"""
Python version: 2.7.6
Version: 1.0 at Sep 01 2014
Author: Chenglong Chen < yr@Kaggle >
Email: c.chenglong@gmail.com
"""

import re
import argparse

def getWeightedVWFile(vwFileTrain, vwWeightedFileTrain, posWeight, negWeight):
    """
    """
    print "     Weight for positive samples: %s" % posWeight
    print "     Weight for negative samples: %s" % negWeight
    # now we write to files
    with open(vwWeightedFileTrain, "wb") as weightedWriter:
        for e,line in enumerate(open(vwFileTrain, "rb")):
            # get the label
            label = int(re.search(r"(^-?[0-9]) '", line).group(1))
            #posWeight = 1
            #negWeight = 13.5
            if label == 1:
                newLine = line[0]+" "+str(posWeight)+line[1:]
            else:
                newLine = line[:2]+" "+str(negWeight)+line[2:]
            weightedWriter.write( newLine )
            if (e+1)%1000000 == 0:
                print "     Wrote %s" % (e+1)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="generate weighted samples")
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("posWeight")
    parser.add_argument("negWeight")
    args = parser.parse_args()
    
    getWeightedVWFile(args.input, args.output,
                      args.posWeight, args.negWeight)