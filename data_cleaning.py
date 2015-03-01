__author__ = 'alicebenziger'

import pandas as pd
import numpy as np
import nltk
from collections import Counter
import string

data_bs_train = pd.read_table("yelp_training_set_business.csv", sep=",")
data_ck_train = pd.read_table("yelp_training_set_checkin.csv", sep=",")
data_rev_train = pd.read_table("yelp_training_set_review.csv", sep=",")
data_user_train = pd.read_table("yelp_training_set_user.csv", sep=",")

data_bs_test = pd.read_table("yelp_test_set_business.csv", sep=",")
data_ck_test = pd.read_table("yelp_test_set_checkin.csv", sep=",")
data_rev_test = pd.read_table("yelp_test_set_review.csv", sep=",")
data_user_test = pd.read_table("yelp_test_set_user.csv", sep=",")

# adding columns row wise for data_checkin
data_ck_train["total_checkins"] = data_ck_train.sum(axis=1)
data_ck_test["total_checkins"] = data_ck_train.sum(axis=1)

#replacing NA's with zeroes in check -in
data_ck_train = data_ck_train.fillna(0)
data_ck_test = data_ck_test.fillna(0)

# new check in dataframe with just total check-in's
data_ck = data_ck_train[["business_id","total_checkins"]]
data_ck_test_new = data_ck_train[["business_id","total_checkins"]]


# freshness of a review
data_rev_train["freshness"] =pd.to_datetime('2013-01-19')- pd.to_datetime(data_rev_train["date"])
data_rev_train["freshness"] = data_rev_train["freshness"]/ np.timedelta64(1, 'D')
# print data_rev_train.head()

data_rev_test["freshness"] = pd.to_datetime('2013-03-12')- pd.to_datetime(data_rev_train["date"])
data_rev_test["freshness"] = data_rev_test["freshness"]/ np.timedelta64(1, 'D')
# print data_rev_test.head()


# features from review text
adj = []
punct = []
data_rev_train["text_length"] = data_rev_train["text"].str.len()
data_rev_train = data_rev_train.fillna(0)

for i in range(data_rev_train.shape[0]):
#for i in range(22740,22741):
    print i
    if data_rev_train["text"][i] == 0:
          adj.append(0)
          punct.append(0)
    else:
        tokens = nltk.tokenize.word_tokenize(data_rev_train["text"][i])
        text =nltk.Text(tokens)
        tags = nltk.tag.pos_tag(text)
        text_feat = Counter([k if k not in string.punctuation else "PUNCT" for k in [j for i, j in tags]])
        adj.append(text_feat['JJ'])
        punct.append(text_feat['PUNCT'])


data_rev_train["count_adj"] = pd.DataFrame(adj)
data_rev_train["count_punct"] = pd.DataFrame(punct)
print data_rev_train.head()


adj_test = []
punct_test = []
data_rev_test["text_length"] = data_rev_test["text"].str.len()
data_rev_test = data_rev_test.fillna(0)

for i in range(data_rev_test.shape[0]):
#for i in range(22740,22741):
    print i
    if data_rev_test["text"][i] == 0:
          adj_test.append(0)
          punct_test.append(0)
    else:
        tokens = nltk.tokenize.word_tokenize(data_rev_test["text"][i])
        text =nltk.Text(tokens)
        tags = nltk.tag.pos_tag(text)
        text_feat = Counter([k if k not in string.punctuation else "PUNCT" for k in [j for i, j in tags]])
        adj.append(text_feat['JJ'])
        punct.append(text_feat['PUNCT'])


data_rev_test["count_adj"] = pd.DataFrame(adj)
data_rev_test["count_punct"] = pd.DataFrame(punct)
print data_rev_test.head()

# create new feature ZIP
ZIP = []
for i in range(data_bs_train.shape[0]):
    ZIP.append(data_bs_train["full_address"][i][-5:])
data_bs_train["ZIP"] = pd.DataFrame(ZIP)
print data_bs_train.head()

ZIP_test = []
for i in range(data_bs_test.shape[0]):
    ZIP_test.append(data_bs_test["full_address"][i][-5:])
data_bs_test["ZIP"] = pd.DataFrame(ZIP_test)
print data_bs_train.head()


#data_merge1 =  pd.merge(data_rev_train,data_bs_train, on ='business_id')
#data_merge2 = pd.merge(data_merge1,data_ck,on='business_id')
#data_train = pd.merge(data_merge2,data_user_train,on = 'user_id')
#data_train.to_csv("train_new.csv")


#data_mergea =  pd.merge(data_rev_test,data_bs_test, on ='business_id')
#data_mergeb = pd.merge(data_mergea,data_ck_test_new,on='business_id')
#data_test= pd.merge(data_mergeb,data_user_test,on = 'user_id')
#data_test.to_csv("test_new.csv")




