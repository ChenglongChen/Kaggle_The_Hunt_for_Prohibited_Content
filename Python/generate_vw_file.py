# -*- coding: utf-8 -*-

"""
Code provided here for constructing features in VW format for:
Kaggle's The Hunt for Prohibited Content Competition
http://www.kaggle.com/c/avito-prohibited-content

Text cleanup majorly follows the provided sample code.

Thank @Triskelion and @Foxtrot for introducing VW.

Python version: 2.7.6
Version: 1.0 at Sep 01 2014
Author: Chenglong Chen < yr@Kaggle >
Email: c.chenglong@gmail.com
"""

######################
## Required Modules ##
######################
import re
import json
import nltk.corpus
import cPickle as pkl
from csv import DictReader
from datetime import datetime
from nltk import SnowballStemmer
from collections import defaultdict
from gensim import corpora, models
from ngram import getUnigram, getBigram

###########
## Setup ##
###########
stopwords= frozenset(word.decode('utf-8') \
                     for word in nltk.corpus.stopwords.words("russian") \
                     if word.decode('utf-8')!="не")
stemmer = SnowballStemmer('russian')
engChars = [ord(char) for char in u"cCyoOBaAKpPeE"]
rusChars = [ord(char) for char in u"сСуоОВаАКрРеЕ"]
eng_rusTranslateTable = dict(zip(engChars, rusChars))
rus_engTranslateTable = dict(zip(rusChars, engChars))

#####################
## Helper function ##
#####################
def tryDivide(x, y):
    """ Try to divide two numbers"""
    s = 0.0
    if y != 0.0: s = x / y
    return s
    
def correctWord (w):
    """ Corrects word by replacing characters with written similarly depending
    on which language the word. 
    Fraudsters use this technique to avoid detection by anti-fraud algorithms.
    """
    if len(re.findall(ur"[а-я]",w))>len(re.findall(ur"[a-z]",w)):
        return w.translate(eng_rusTranslateTable)
    else:
        return w.translate(rus_engTranslateTable)
        
def getWordCharCount(w):
    """ Char count for a word."""
    rus = len(re.findall(ur"[а-я]",w))
    eng = len(re.findall(ur"[a-z]",w))
    c = len(w)    
    return c, rus, eng
    
def getTextStatsFeat(text, stemmRequired = True,
                     excludeStopwordsRequired = True):
    """ Get stats features for raw text.
    These features don't seem to help much.
    """
    #length = len(text)
    sentenceCount = len(re.findall("[.?!]", text))
    exclamationMarkCount = len(re.findall("[!]", text))
    questionMarkCount = len(re.findall("[?]", text))
    digitsCount = len(re.findall("[0-9]+", text))
    text = text.replace(",", " ").replace(".", " ")
    cleanText = re.sub(u'[^a-zа-я0-9]', ' ', text.lower())
    wordCount = 0.0
    charCount = 0.0
    rusCharCount = 0.0
    engCharCount = 0.0
    if excludeStopwordsRequired:
        for w in cleanText.split():
            if len(w)>1 and w not in stopwords:
                if not (not stemmRequired or re.search("[0-9a-z]", w)):
                    w = stemmer.stem(w)
                wordCount += 1
                c, rus, eng = getWordCharCount(w)
                charCount += c
                rusCharCount += rus
                engCharCount += eng
    else:
        for w in cleanText.split():
            if len(w)>1:
                if not (not stemmRequired or re.search("[0-9a-z]", w)):
                    w = stemmer.stem(w)
                wordCount += 1
                c, rus, eng = getWordCharCount(w)
                charCount += c
                rusCharCount += rus
                engCharCount += eng
    # per sentence
    wordPerSentence = tryDivide(wordCount, sentenceCount)
    charPerSentence = tryDivide(charCount, sentenceCount)
    rusCharPerSentence = tryDivide(rusCharCount, sentenceCount)
    engCharPerSentence = tryDivide(engCharCount, sentenceCount)
    # per word
    charPerWord = tryDivide(charCount, wordCount)
    rusCharPerWord = tryDivide(rusCharCount, wordCount)
    engCharPerWord = tryDivide(engCharCount, wordCount)
    # ratio
    rusCharRatio = tryDivide(rusCharCount, charCount)
    engCharRatio = tryDivide(engCharCount, charCount)
    rusCharVsEngChar = tryDivide(rusCharCount, engCharCount)
    engCharVsRusChar = tryDivide(engCharCount, rusCharCount)
    
    stats = [
    sentenceCount,
    wordCount,
    charCount,
    rusCharCount,
    engCharCount,
    digitsCount,
    exclamationMarkCount,
    questionMarkCount,
    wordPerSentence,
    charPerSentence,
    rusCharPerSentence,
    engCharPerSentence,
    charPerWord,
    rusCharPerWord,
    engCharPerWord,
    rusCharRatio,
    engCharRatio,
    rusCharVsEngChar,
    engCharVsRusChar,
    ]
    statsFeat = ""
    for i,f in enumerate(stats):
        if f != 0:
            statsFeat += "%s:%s " % (i+1, f)
    statsFeat = statsFeat[:-1]    
    return statsFeat
    
def getWords(text, stemmRequired = True,
             correctWordRequired = True,
             excludeStopwordsRequired = True):
    """ Splits the text into words, discards stop words and applies stemmer. 
    Parameters
    ----------
    text : str - initial string
    stemmRequired : bool - flag whether stemming required
    correctWordRequired : bool - flag whether correction of words required     
    """
    text = text.replace(",", " ").replace(".", " ")
    cleanText = re.sub(u'[^a-zа-я0-9]', ' ', text.lower())
    if correctWordRequired:
        if excludeStopwordsRequired:
            words = [correctWord(w) \
                    if not stemmRequired or re.search("[0-9a-z]", w) \
                    else stemmer.stem(correctWord(w)) \
                    for w in cleanText.split() \
                    if len(w)>1 and w not in stopwords]
        else:
            words = [correctWord(w) \
                    if not stemmRequired or re.search("[0-9a-z]", w) \
                    else stemmer.stem(correctWord(w)) \
                    for w in cleanText.split() \
                    if len(w)>1]
    else:
        if excludeStopwordsRequired:
            words = [w \
                    if not stemmRequired or re.search("[0-9a-z]", w) \
                    else stemmer.stem(w) \
                    for w in cleanText.split() \
                    if len(w)>1 and w not in stopwords]
        else:
            words = [w \
                    if not stemmRequired or re.search("[0-9a-z]", w) \
                    else stemmer.stem(w) \
                    for w in cleanText.split() \
                    if len(w)>1]
    
    return words
    
def getAttrsDict(attrs):
    """ Clean attributes in json format."""
    attrsDict = json.loads(re.sub('/\"(?!(,\s"|}))','\\"',attrs).replace("\t"," ").replace("\n"," ")) if len(attrs)>0 else {}
    return attrsDict
    
def setDefaultCount():
    return 0
    
def setDefaultIndex():
    """See the following why using this method for dumping defaultdict
    http://stackoverflow.com/questions/16439301/cant-pickle-defaultdict
    http://bytes.com/topic/python/answers/785634-problem-pickle-collections
    -defaultdict-default_factory-set-do-not-work
    """
    return 0
    
def getDataIndex(tsvFile):
    """ Get index dict for category, subcategory, attrs key & value,
    and word in title and description.
    You only need to run this once.
    """
    start = datetime.now()
    
    # dict for counts
    catCounts = defaultdict(setDefaultCount)
    subCatCounts = defaultdict(setDefaultCount)
    attrsKeyCounts = defaultdict(setDefaultCount)
    attrsValCounts = defaultdict(setDefaultCount)
    wordCounts = defaultdict(setDefaultCount)
    # dict for index
    catIndex = defaultdict(setDefaultIndex)
    subCatIndex = defaultdict(setDefaultIndex)
    attrsKeyIndex = defaultdict(setDefaultIndex) 
    attrsValIndex = defaultdict(setDefaultIndex)
    wordIndex = defaultdict(setDefaultIndex)
    
    with open(tsvFile, "rb") as tsvReader:
        itemReader = DictReader(tsvReader, delimiter='\t', quotechar='"')
        for i, item in enumerate(itemReader):
            item = {featureName:featureValue.decode('utf-8') \
                    for featureName,featureValue in item.iteritems() \
                    if featureValue is not None}
                        
            catCounts[item["category"]] += 1
            subCatCounts[item["subcategory"]] += 1
            
            if item.has_key("attrs"):
                attrsDict = getAttrsDict(item["attrs"])
                for k,v in attrsDict.items():
                    attrsKeyCounts[k] += 1
                    attrsValCounts[v] += 1
       
            s = item["title"] + " " + item["description"]
            for word in getWords(s):
                wordCounts[word] += 1
                    
            if (i+1)%10000 == 0:
                print( "\n%s\t%s"%((i+1),str(datetime.now() - start)) )
                print( "Counts for category:" )
                print( catCounts.values()[:5] )
                print( "Counts for subcategory:" )
                print( subCatCounts.values()[:5] )
                print( "Counts for keys of attrs:" )
                print( attrsKeyCounts.values()[:5] )
                print( "Counts for values of attrs:" )
                print( attrsValCounts.values()[:5] )
                print( "Counts for words in title and description:" )
                print( wordCounts.values()[:5] )
                    
    # get the index
    for index, (cat, count) in enumerate(catCounts.iteritems()):
        catIndex[cat] = index+1

    for index, (subCat, count) in enumerate(subCatCounts.iteritems()):
        subCatIndex[subCat] = index+1

    for index, (attrsKey, count) in enumerate(attrsKeyCounts.iteritems()):
        attrsKeyIndex[attrsKey] = index+1

    for index, (attrsVal, count) in enumerate(attrsValCounts.iteritems()):
        attrsValIndex[attrsVal] = index+1

    for index, (word, count) in enumerate(wordCounts.iteritems()):
        wordIndex[word] = index+1
     
    dataIndex = {"catIndex": catIndex,
                 "subCatIndex": subCatIndex,
                 "attrsKeyIndex": attrsKeyIndex,
                 "attrsValIndex": attrsValIndex,
                 "wordIndex": wordIndex}
    return dataIndex           

def getRelevantIDsAndProvedIDs(tsvFile):
    """ Get all relevant IDs.
    You only need to run this once.
    """    
    start = datetime.now()
    relevantIDs = []
    provedIDs = []
    with open(tsvFile, "rb") as tsvReader:
        itemReader = DictReader(tsvReader, delimiter='\t', quotechar='"')
        for i, item in enumerate(itemReader):
            item = {featureName:featureValue.decode('utf-8') \
                    for featureName,featureValue in item.iteritems() \
                    if featureValue is not None}
            
            if item["is_blocked"] == "1":
                relevantIDs.append( int(item["itemid"]) )
                if item["is_proved"] == "1":
                    provedIDs.append( int(item["itemid"]) ) 

            if (i+1)%1000000 == 0:
                print( "%s\t%s"%((i+1),str(datetime.now() - start)) )

    return relevantIDs, provedIDs
    
class DescriptionReader(object):
    """ Reader to read description in memory friendly streaming style.
    Return 1/2gram of the text. You can extend it to read title or attrs.
    """
    def __init__(self, tsvFile, wordIndex, ngram):
        self.tsvFile = tsvFile
        self.wordIndex = wordIndex
        self.ngram = ngram
        self.counter = 0
    def __iter__(self):
        for item in DictReader(open(self.tsvFile, "rb"), delimiter='\t', quotechar='"'):
            self.counter += 1
            item = {featureName:featureValue.decode('utf-8') \
                    for featureName,featureValue in item.iteritems() \
                    if featureValue is not None}
            description = [ str(self.wordIndex[w]) for w in getWords(item["description"]) ]
            if self.ngram == 1:
                yield getUnigram(description)
            elif self.ngram == 2:
                yield getBigram(description, "_")
            if self.counter%100000 == 0:
                print( "     Process %s" % self.counter )
                
def testDescriptionReader(tsvFile, wordIndex, ngram):
    """ Test DescriptionReader."""
    for d in DescriptionReader(tsvFile, wordIndex, ngram):
        print d
        break
    
def getDictionary(tsvFile, wordIndex, ngram):
    """ Get dictionary from the training corpus."""
       
    reader = DescriptionReader(tsvFile, wordIndex, ngram)
    dictionary = corpora.Dictionary( d for d in reader )

    # remove stop words and words that appear only once
    stoplist = [] # might be specified in the furture
    stop_ids = [dictionary.token2id[stopword] for stopword in stoplist
                 if stopword in dictionary.token2id]
    once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq == 1]
    dictionary.filter_tokens(stop_ids + once_ids) # remove stop words and words that appear only once
    dictionary.compactify() # remove gaps in id sequence after words that were removed
    
    return dictionary
    
def trainTfidfModel(tsvFile, wordIndex, ngram, dictionary):
    """ Train tf-idf model"""
    reader = DescriptionReader(tsvFile, wordIndex, ngram)
    # construct the dictionary one query at a time
    tfidf_model = models.tfidfmodel.TfidfModel( dictionary.doc2bow(d) for d in reader )
    
    return tfidf_model
         
def getTfidfFeat(words, dictionary, tfidf_model):
    """ Generate tf-idf feature in VW format."""
    # bow feats
    bow = dictionary.doc2bow(words)
    # tf-idf feats
    tfidf = tfidf_model[bow]
    feat = ""
    if len(tfidf) > 0:
        for f in tfidf:
            feat += "%s:%s " % (f[0], f[1])
        feat = feat[:-1]
    return feat
         
def getVWFile(tsvFile, vwFile, dataIndex, dictionary, tfidf_model, train=True):
    """ Generate the overall features in VW format."""
    start = datetime.now()
    
    # extract all the index
    catIndex = dataIndex["catIndex"]
    subCatIndex = dataIndex["subCatIndex"]
    attrsKeyIndex = dataIndex["attrsKeyIndex"]
    attrsValIndex = dataIndex["attrsValIndex"]
    wordIndex = dataIndex["wordIndex"]
    
    with open(vwFile, "wb") as vwWriter:
        with open(tsvFile, "rb") as tsvReader:
            itemReader = DictReader(tsvReader, delimiter='\t', quotechar='"')
            for i, item in enumerate(itemReader):
                item = {featureName:featureValue.decode('utf-8') \
                        for featureName,featureValue in item.iteritems() \
                        if featureValue is not None}

                # get header
                itemid = int(item["itemid"])
                label = int(item["is_blocked"]) if train else 1
                header = "%s '%s " % (int(2*label - 1), itemid)
                
                # category
                categoryFeat = "|C %s " % catIndex[ item["category"] ]
                
                # subcategory
                subcategoryFeat = "|SC %s " % subCatIndex[ item["subcategory"] ]
                
                # title
                title = [ str(wordIndex[w]) for w in getWords(item["title"]) ]
                # first-gram
                title_start = title[0] if len(title)>0 else "0"
                # end-gram
                title_end = title[-1] if len(title)>0 else "0"
                # naming is a pain for me
                titleFeat = "|T %s |bT %s |cT %s " % (" ".join(title), title_start, title_end)
                titleStatsFeat = "|t %s " % getTextStatsFeat(item["title"])
                
                # description
                description = [ str(wordIndex[w]) for w in getWords(item["description"]) ]
                # first-gram
                description_start = description[0] if len(description)>0 else "0"
                # end-gram
                description_end = description[-1] if len(description)>0 else "0"
                descriptionFeat = "|D %s |fD %s |gD %s " % (" ".join(description), description_start, description_end)
                descriptionStatsFeat = "|d %s " % getTextStatsFeat(item["description"])
                tfidf_feat1 = getTfidfFeat(getUnigram(description), dictionary[1], tfidf_model[1])
                # 2gram tfidf seem to harm the performance, you are save to drop it here
                tfidf_feat2 = getTfidfFeat(getBigram(description, "_"), dictionary[2], tfidf_model[2])
                descriptionFeat += "|iD %s |jD %s " % (tfidf_feat1, tfidf_feat2)
                
                # attrs
                attrsFeat = ""
                countAttrs = 0
                if item.has_key("attrs"):
                    attrsDict = getAttrsDict(item["attrs"])
                    #print attrs
                    for k,v in attrsDict.items():
                        countAttrs += 1
                        attrsFeat += "|A%s %s " % (attrsKeyIndex[k], attrsValIndex[v])
                    attrsFeat += "|a "
                    for k,v in attrsDict.items():
                        attrsFeat += "%s " % (attrsKeyIndex[k])                        
                if len(attrsFeat) == 0:
                    attrsFeat = "|NA 1 "
                attrsFeat += "|hAC %s " % countAttrs
                
                # price
                priceFeat = "|P %s " % item["price"]                
                # phones_cnt
                phonesCntFeat = "|p %s " % item["phones_cnt"]
                # emails_cnt
                emailsCntFeat = "|e %s " % item["emails_cnt"]
                # urls_cnt
                urlsCntFeat = "|u %s " % item["urls_cnt"]
                
                # output
                vwLine = header \
                       + categoryFeat \
                       + subcategoryFeat \
                       + titleFeat \
                       + titleStatsFeat \
                       + descriptionFeat \
                       + descriptionStatsFeat \
                       + attrsFeat \
                       + priceFeat \
                       + phonesCntFeat \
                       + emailsCntFeat \
                       + urlsCntFeat[:-1] + "\n"
                vwWriter.write( vwLine )
                
                # report progress
                if (i+1)%10000 == 0:
                    print( "\n%s\t%s"%((i+1),str(datetime.now() - start)) )
                    print( "Sample output:\n%s" % vwLine )
                    
def getProvedVWFile(vwFileTrain, vwProvedFileTrain,
                    vwUnprovedFileTrain, provedIDs):
    """ Seperate the trianing data into proved and unproved data.
    """
    # convert to dict for faster checking whether an id is in provedIDs
    provedIDsDict = dict()
    for id in provedIDs:
        provedIDsDict[id] = 1
    # now we write to files
    with open(vwProvedFileTrain, "wb") as provedWriter:
        with open(vwUnprovedFileTrain, "wb") as unProvedWriter:
            for e,line in enumerate(open(vwFileTrain, "rb")):
                # get the label
                label = int(re.search(r"(^-?[0-9]) '", line).group(1))
                # get the ID
                id = int(re.search(r"'([0-9]+)", line).group(1))
                if label == -1:
                    # write unblocked samples to both files
                    provedWriter.write( line )
                    unProvedWriter.write( line )
                else:
                    if provedIDsDict.has_key(id):
                        provedWriter.write( line )
                    else:
                        unProvedWriter.write( line )
                if (e+1)%1000000 == 0:
                    print "     Wrote %s" % (e+1)

def getWeightedVWFile(vwFileTrain, vwWeightedFileTrain, posWeight, negWeight):
    """ Generate weighted training data.
    """
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
                
##########
## Main ##
##########
def main():

    dataPath = "../Data/"
    tsvFileTrain = dataPath + "avito_train.tsv"
    tsvFileTest = dataPath + "avito_test.tsv"
    
    vwFileTrain = dataPath + "train.vw"
    vwProvedFileTrain = dataPath + "train_proved.vw"
    vwUnprovedFileTrain = dataPath + "train_unproved.vw"
    vwFileTest = dataPath + "test.vw"
    
#    vwWeightedFileTrain = dataPath + "train_weighted_Aug18.vw"
    
    dataIndexFile = dataPath + "dataIndex.pkl"
    IDsFile = dataPath + "relevantIDsAndProvedIDs.pkl"
    tfidfModelFile = dataPath + "tfidf_model.pkl"
    
    #####################
    ## Load data index ##
    #####################
    # load data index that is used for mapping all categorical variables including
    # tokens in the title and description to integer
    try:
        with open(dataIndexFile, "rb") as f:
            dataIndex = pkl.load(f)
    except:
        dataIndex = getDataIndex(tsvFileTrain)
        with open(dataIndexFile, "wb") as f:
            pkl.dump(dataIndex, f, -1)
            
    ##############
    ## Load IDs ##
    ##############
    # load relevant IDs (for computing valid AP@k) and proved blocked IDs
    try:
        with open(IDsFile, "rb") as f:
            relevantIDs, provedIDs = pkl.load(f)
    except:
        relevantIDs, provedIDs = getRelevantIDsAndProvedIDs(tsvFileTrain)
        with open(IDsFile, "wb") as f:
            pkl.dump((relevantIDs,provedIDs), f, -1)
        
    #######################
    ## Load Tf-Idf model ##
    #######################
    try:
        with open(tfidfModelFile, "rb") as f:
            dictionary, tfidf_model = pkl.load(f)
    except:
        ngrams = [1, 2]
        dictionary = dict()
        tfidf_model = dict()
        for ngram in ngrams:
            testDescriptionReader(tsvFileTrain, dataIndex["wordIndex"], ngram)
            print "Get the dictionary for %s-gram" % ngram
            dictionary[ngram] = getDictionary(tsvFileTrain, dataIndex["wordIndex"], ngram)
            print "Train tf-idf model for %s-gram" % ngram
            tfidf_model[ngram] = trainTfidfModel(tsvFileTrain, dataIndex["wordIndex"], ngram, dictionary[ngram])
    
        # save the models
        with open(tfidfModelFile, 'wb') as f:
            pkl.dump((dictionary, tfidf_model), f, -1)
            
    ############################
    ## Generate features file ##
    ############################
    getVWFile(tsvFileTrain, vwFileTrain, dataIndex, dictionary, tfidf_model, train=True)
    getVWFile(tsvFileTest, vwFileTest, dataIndex, dictionary, tfidf_model, train=False)
    getProvedVWFile(vwFileTrain, vwProvedFileTrain, vwUnprovedFileTrain, provedIDs)
#    posWeight = 1
#    negWeight = 15
#    getWeightedVWFile(vwFileTrain, vwWeightedFileTrain, posWeight, negWeight)

                       
if __name__=="__main__":
    main()