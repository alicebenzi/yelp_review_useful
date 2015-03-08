__author__ = 'alicebenziger'

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import MiniBatchKMeans




def label_encoder(data, binary_cols):
    label_enc = LabelEncoder()
    # for col in categorical_cols:
    #     label_enc.fit(data[col])
    #     data[col] = label_enc.transform(data[col])
    # encoded_categorical = np.array(data[categorical_cols])

    for col in binary_cols:
        label_enc.fit(data[col])
        data[col] = label_enc.transform(data[col])
    encoded_binary = np.array(data[binary_cols])

    return encoded_binary


def dummy_encoder(train_X,test_X,categorical_variable_list):
    enc = OneHotEncoder(n_values ='auto',categorical_features=categorical_variable_list)
    # train_X = enc.fit_transform(train_X)
    # test_X = enc.transform(test_X)

    #
    train_X = enc.fit_transform(train_X).toarray()
    test_X = enc.transform(test_X).toarray()
    return train_X, test_X



def normalize(X):
    normalizer = preprocessing.Normalizer().fit(X)
    normalized_X = normalizer.transform(X)
    return normalized_X


def category_manipulation(train,test):
    vect = CountVectorizer(tokenizer=lambda text: text.split(','))

    cat_fea = vect.fit_transform(train['categories'].fillna(''))
    cat_fea = cat_fea.todense()
    idx_max_1 = cat_fea > 1
    cat_fea[idx_max_1] = 1

    cat_fea_test = vect.transform(test['categories'].fillna(''))
    cat_fea_test = cat_fea_test.todense()
    idx_max_1 = cat_fea_test > 1
    cat_fea_test[idx_max_1] = 1

    # CATEGORY CLUSTERS
    #  Based on the category extracted before, the idea is to create a n clusters to
    #  aggregate set of similar categories
    # for esti in (2,3):#,60,70,80,90,100,110,125):
    esti = 50
    km = MiniBatchKMeans(n_clusters=esti, random_state=888)#, init_size=esti*10)
    #       init='k-means++', n_clusters=3, n_init=10
    km.fit(cat_fea)

    # train['cat_clust_'+str(esti)] = km.predict(cat_fea)
    # test['cat_clust_'+str(esti)] = km.predict(cat_fea_test)
    train['cat_clust_100'] = km.predict(cat_fea)
    test['cat_clust_100'] = km.predict(cat_fea_test)
    # return train,test


def train_test():

    train = pd.read_csv('train_new.csv', header=0)
    pred = pd.read_csv('test_new.csv', header=0)

    print train.head(), pred.head()



    category_manipulation(train, pred)

    review_id = pred["review_id"]

    #creating new variables

    # train["stars_per_checkin"] = train["stars_bs"]/(train["total_checkins"]+1)
    # pred["stars_per_checkin"] = pred["stars_bs"]/(pred["total_checkins"]+1)


    del train["categories"], pred["categories"], train["review_id"], pred["review_id"],train["count_punct"], \
        pred["count_punct"],train["count_adj"], pred["count_adj"], \
        train["freshness"], pred["freshness"], train["stars_bs"], pred["stars_bs"],\
        train["total_checkins"], pred["total_checkins"]
        # train["cat_clust_100"], pred["cat_clust_100"], # train["zip_code"], \
        # pred["zip_code"]#, train["open"], pred["open"], train["total_checkins"], pred["total_checkins"], \
        # train["count_punct"], pred["count_punct"], train["count_adj"], pred["count_adj"]





    target = np.array(train["votes_useful_rev"])
    target_var = ["votes_useful_rev"]
    binary_var = ["open"]
    categorical_var = ["cat_clust_100", "zip_code"]
    # categorical_var = ["zip_code"]

    col_names_train= train.columns.values
    col_names_pred = pred.columns.values
    # print col_names_train
    numerical_var = [col for col in col_names_train if col not in binary_var if col not in categorical_var if col not in target_var]
    numerical_var_pred = [col for col in col_names_pred if col not in binary_var if col not in categorical_var]
    # numerical_var = [col for col in col_names_train  if col not in categorical_var if col not in target_var]
    # numerical_var_pred = [col for col in col_names_pred  if col not in categorical_var]
    encoded_binary = label_encoder(train, binary_var)
    encoded_binary_pred= label_encoder(pred, binary_var)
    categorical_variables = np.array(train[categorical_var])
    categorical_variables_pred = np.array(pred[categorical_var])
    numerical_data = np.array(train[numerical_var])
    numerical_data_pred = np.array(pred[numerical_var_pred])
    #
    #
    #
    training_data_transformed = np.concatenate((categorical_variables,encoded_binary,numerical_data),axis=1)
    pred_data_transformed = np.concatenate((categorical_variables_pred,encoded_binary_pred,numerical_data_pred),axis=1)
    # training_data_transformed = np.concatenate((categorical_variables,numerical_data),axis=1)
    # pred_data_transformed = np.concatenate((categorical_variables_pred,numerical_data_pred),axis=1)
    #
    #
    #
    train_x = training_data_transformed
    train_y = target
    pred_x = pred_data_transformed
    #
    # print len(train_x[0]), len(pred_x[0])
    #
    pd.DataFrame(train_x).to_csv("test_labelenc.csv")

    train_x, pred_x = dummy_encoder(train_x, pred_x, categorical_variable_list=list(range(0,1)))
    #
    #normalizing (if required)
    train_x_norm = normalize(train_x)
    pred_x_norm = normalize(pred_x)
    # print train_x[0:1,:]
    #
    #
    print "done.."
    return train_x, train_y, pred_x, train_x_norm, pred_x_norm, review_id


