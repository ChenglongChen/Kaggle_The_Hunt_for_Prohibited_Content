# -*- coding: UTF-8 -*-

"""
Python version: 2.7.6
Version: 1.0 at Sep 01 2014
Author: Chenglong Chen < yr@Kaggle >
Email: c.chenglong@gmail.com
"""

def getUnigram(words):
    """
    Input: a list of words, e.g., ['I', 'am', 'Denny']
    Output: a list of unigram
    """
    assert type(words) == list
    return words
    
def getBigram(words, join_string):
    """
    Input: a list of words, e.g., ['I', 'am', 'Denny']
    Output: a list of bigram, e.g., ['I_am', 'am_Denny']
    I use _ as join_string for this example.
    """
    assert type(words) == list
    L = len(words)
    if L > 1:
        lst = []
        for i in xrange(L-1):
            lst.append( join_string.join([words[i], words[i+1]]) )
    else:
        # set it as unigram
        lst = getUnigram(words)
    return lst
    
def getTrigram(words, join_string):
    """
    Input: a list of words, e.g., ['I', 'am', 'Denny']
    Output: a list of trigram, e.g., ['I_am_Denny']
    I use _ as join_string for this example.
    """
    assert type(words) == list
    L = len(words)
    if L > 2:
        lst = []
        for i in xrange(L-2):
            lst.append( join_string.join([words[i], words[i+1], words[i+2]]) )
    else:
        # set it as bigram
        lst = getBigram(words, join_string)
    return lst
    
def getFourgram(words, join_string):
    """
    Input: a list of words, e.g., ['I', 'am', 'Denny', 'boy']
    Output: a list of trigram, e.g., ['I_am_Denny_boy']
    I use _ as join_string for this example.
    """
    assert type(words) == list
    L = len(words)
    if L > 3:
        lst = []
        for i in xrange(L-3):
            lst.append( join_string.join([words[i], words[i+1], words[i+2], words[i+3]]) )
    else:
        # set it as bigram
        lst = getTrigram(words, join_string)
    return lst

def getBiterm(words, join_string):
    """
    Input: a list of words, e.g., ['I', 'am', 'Denny']
    Output: a list of biterm, e.g., ['I_am', 'I_Denny', 'am_I', 'am_Denny', 'Denny_I', 'Denny_am']
    I use _ as join_string for this example.
    """
    assert type(words) == list
    L = len(words)
    if L > 1:
        lst = []
        for i in xrange(L):
            for j in xrange(L):
                if j != i:
                    lst.append( join_string.join([words[i], words[j]]) )
    else:
        # set it as unigram
        lst = getUnigram(words)
    return lst
    
def getTriterm(words, join_string):
    """
    Input: a list of words, e.g., ['I', 'am', 'Denny']
    Output: a list of triterm, e.g., ['I_am_Denny', 'I_Denny_am', 'am_I_Denny',
    'am_Denny_I', 'Denny_I_am', 'Denny_am_I']
    I use _ as join_string for this example.
    """
    assert type(words) == list
    L = len(words)
    if L > 2:
        lst = []
        for i in xrange(L):
            for j in xrange(L):
                if j != i:
                    for k in xrange(L):
                        if k != i and k != j:
                            lst.append( join_string.join([words[i], words[j], words[k]]) )
    else:
        # set it as biterm
        lst = getBiterm(words, join_string)
    return lst