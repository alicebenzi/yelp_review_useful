__author__ = 'alicebenziger'

import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
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
data_ck_test_new = data_ck_test[["business_id","total_checkins"]]


# freshness of a review
data_rev_train["freshness"] =pd.to_datetime('2013-01-19')- pd.to_datetime(data_rev_train["date"])
data_rev_train["freshness"] = data_rev_train["freshness"]/ np.timedelta64(1, 'D')
# print data_rev_train.head()

data_rev_test["freshness"] = pd.to_datetime('2013-03-12')- pd.to_datetime(data_rev_train["date"])
data_rev_test["freshness"] = data_rev_test["freshness"]/ np.timedelta64(1, 'D')
# print data_rev_test.head()


# features from review text
# adj = []
# punct = []
# count_sent = []
# data_rev_train["text_length"] = data_rev_train["text"].str.len()
# data_rev_train = data_rev_train.fillna(0)
#
# for i in range(data_rev_train.shape[0]):
#     #print i
#     if data_rev_train["text"][i] == 0:
#           adj.append(0)
#           punct.append(0)
#     else:
#
#         tokens = nltk.tokenize.word_tokenize(data_rev_train["text"][i])
#         text =nltk.Text(tokens)
#         sentences = nltk.tokenize.sent_tokenize(data_rev_train["text"][i])
#         count_sent.append(len(sentences))
#         tags = nltk.tag.pos_tag(text)
#         text_feat = Counter([k if k not in string.punctuation else "PUNCT" for k in [j for i, j in tags]])
#         adj.append(text_feat['JJ'])
#         punct.append(text_feat['PUNCT'])
#
#
# data_rev_train["count_adj"] = pd.DataFrame(adj)
# data_rev_train["count_punct"] = pd.DataFrame(punct)
# data_rev_train["count_sentences"] = pd.DataFrame(count_sent)
# #print data_rev_train.head()
#
#
# adj_test = []
# punct_test = []
# count_sent_test = []
# data_rev_test["text_length"] = data_rev_test["text"].str.len()
# data_rev_test = data_rev_test.fillna(0)
#
# for i in range(data_rev_test.shape[0]):
#     #print i
#     if data_rev_test["text"][i] == 0:
#           adj_test.append(0)
#           punct_test.append(0)
#     else:
#         tokens = nltk.tokenize.word_tokenize(data_rev_test["text"][i])
#         text =nltk.Text(tokens)
#         sentences = nltk.tokenize.sent_tokenize(data_rev_train["text"][i])
#         count_sent_test.append(len(sentences))
#         tags = nltk.tag.pos_tag(text)
#         text_feat = Counter([k if k not in string.punctuation else "PUNCT" for k in [j for i, j in tags]])
#         adj.append(text_feat['JJ'])
#         punct.append(text_feat['PUNCT'])
#
#
# data_rev_test["count_adj"] = pd.DataFrame(adj)
# data_rev_test["count_punct"] = pd.DataFrame(punct)
# data_rev_test["count_sentences"] = pd.DataFrame(count_sent_test)

#print data_rev_test.head()

# data_rev_train.to_csv("yelp_rev_train_feat.csv", index=False)
# data_rev_test.to_csv("yelp_rev_tst_feat.csv", index=False)


# create new feature ZIP
ZIP = []
for i in range(data_bs_train.shape[0]):
    ZIP.append(data_bs_train["full_address"][i][-5:])
data_bs_train["zip_code"] = pd.DataFrame(ZIP)
# print data_bs_train.head()

ZIP_test = []
for i in range(data_bs_test.shape[0]):
    ZIP_test.append(data_bs_test["full_address"][i][-5:])
data_bs_test["zip_code"] = pd.DataFrame(ZIP_test)
# print data_bs_train.head()


# categories cleaning on train data

# all_categories_yelp = pd.read_table('yelp-business-categories-list/Sheet2-Table 1.csv',sep=",",header=None)
# all_categories_yelp = all_categories_yelp[2]
# l1 = list(all_categories_yelp)
# l1_set = set(l1)

#print data_bs_train.head()
all_categories = data_bs_train['categories']
all_categories[pd.isnull(all_categories)] = 'Other'
data_bs_train['categories'] = all_categories
#print data_bs_train.head()
#
#
# for i in range(0,len(data_bs_train)):
#     a = data_bs_train['categories'][i]
#     a_s = a.split(',')
#     a_set = set(a_s)
#     if not l1_set.intersection(a_set):
#         #count += 1
#         a_set = 'Other'


## categories cleaning on test data
all_categories = data_bs_test['categories']
all_categories[pd.isnull(all_categories)] = 'Other'
data_bs_test['categories'] = all_categories

# for i in range(0,len(data_bs_test)):
#     a = data_bs_test['categories'][i]
#     a_s = a.split(',')
#     a_set = set(a_s)
#     if not l1_set.intersection(a_set):
#         #count += 1
#         a_set = 'Other'




# create user similarity clusters based on review count and average stars from review table.
data_user_train_temp = data_user_train[["average_stars","review_count"]]
data_user_test_temp = data_user_test[["average_stars","review_count"]]
# apply kmeans to create clusters
k = 125
kmeans = MiniBatchKMeans(n_clusters=k, random_state=1377, init_size=k*10)
kmeans.fit(data_user_train_temp)
# create column of similar users in user table in training set.
data_user_train['user_id_cluster_'+str(k)] = kmeans.predict(data_user_train_temp)
# create column of similar users in user table in test set.
data_user_test['user_id_cluster_'+str(k)] = kmeans.predict(data_user_test_temp)


data_rev_train_1 = pd.read_csv("yelp_rev_train_feat.csv",sep=',')
data_rev_test_1 = pd.read_csv("yelp_rev_tst_feat.csv", sep =',')



data_merge1 =  pd.merge(data_rev_train_1,data_bs_train, on ='business_id')
data_merge2 = pd.merge(data_merge1,data_ck,on='business_id')
data_train = pd.merge(data_merge2,data_user_train,on = 'user_id')
data_train.to_csv("train_new.csv",index=False)



data_mergea =  pd.merge(data_rev_test_1,data_bs_test, on ='business_id')
data_mergeb = pd.merge(data_mergea,data_ck_test_new,on='business_id')
data_test= pd.merge(data_mergeb,data_user_test,on = 'user_id')
data_test.to_csv("test_new.csv", index=False)



