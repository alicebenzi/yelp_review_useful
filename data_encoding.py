__author__ = 'alicebenziger'
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import CountVectorizer




# def label_encoder(data, binary_cols, categorical_cols):
#     label_enc = LabelEncoder()
#     for col in categorical_cols:
#         label_enc.fit(data[col])
#         data[col] = label_enc.transform(data[col])
#     encoded_categorical = np.array(data[categorical_cols])
#
#     for col in binary_cols:
#         label_enc.fit(data[col])
#         data[col] = label_enc.transform(data[col])
#     encoded_binary = np.array(data[binary_cols])
#     return encoded_categorical, encoded_binary

def label_encoder(data, binary_cols):
    label_enc = LabelEncoder()

    for col in binary_cols:
        label_enc.fit(data[col])
        data[col] = label_enc.transform(data[col])
    encoded_binary = np.array(data[binary_cols])
    return  encoded_binary


def dummy_encoder(train_X,test_X,categorical_variable_list):
    enc = OneHotEncoder(n_values ='auto',categorical_features=categorical_variable_list)
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
    
    ## CATEGORY CLUSTERS
    #  Based on the category extracted before, the idea is to create a n clusters to
    #  aggregate set of similar categories
    for esti in (20,35,50,60,70,80,90,100,110,125):
        km = KMeans(n_clusters=esti, random_state=888)#, init_size=esti*10)
        #         init='k-means++', n_clusters=3, n_init=10
        print "fitting "+str(esti)+" clusters - category"
        init_time = time.time()
        km.fit(cat_fea)
        print (time.time()-init_time)/60
        
        train['cat_clust_'+str(esti)] = km.predict(cat_fea)
        test['cat_clust_'+str(esti)] = km.predict(cat_fea_test)
    
    return train,test


def train_test():

    train = pd.read_csv('train_new.csv', header=0)
    pred = pd.read_csv('test_new.csv', header=0)
    
    train, pred = category_manipulation(train, pred)

    # print train.columns.values
    bus_id = pred['business_id']
    

    del train["business_id"], train["date"], train["review_id"], train["text"], train["type_x"], train["user_id"],\
        train["votes_cool_x"], train["votes_funny_x"],train["full_address"], train["latitude"], train["longitude"],\
        train["name_x"], train["neighborhoods"], train["type_y"], train["name_y"], train["type"], train["votes_cool_y"],\
        train["votes_funny_y"], train["votes_useful_y"], train["state"], train["city"],train["category"]

    #print test.columns.values
    del pred["business_id"], pred["date"], pred["review_id"], pred["text"], pred["type_x"], pred["user_id"],\
        pred["full_address"], pred["latitude"], pred["longitude"],\
        pred["name_x"], pred["neighborhoods"], pred["type_y"], pred["name_y"], pred["type"], pred["state"], pred["city"],pred["category"]




    ### Vectorizing  ZIP
    vect = CountVectorizer(tokenizer=lambda text: text.split(','))
    zip_fea = vect.fit_transform(train['zip_code'])
    print zip_fea
    zip_fea = zip_fea.todense()
    print zip_fea
    idx_max_1 = zip_fea > 1
    zip_fea[idx_max_1] = 1
    zip_fea_test = vect.transform(pred['zip_code'])
    zip_fea_test = zip_fea_test.todense()
    idx_max_1 = zip_fea_test > 1
    zip_fea_test[idx_max_1] = 1


    km = MiniBatchKMeans(n_clusters=100, random_state=1377, init_size=100*10)

    km.fit(zip_fea)
    train['zip_clust'] = km.predict(zip_fea)
    pred['zip_clust'] = km.predict(zip_fea_test)

    #deleting actual zip codes
    del train['zip_code'], pred['zip_code']



    print train.head()
    print pred.head()

    target = np.array(train["votes_useful_x"])
    target_var = ["votes_useful_x"]
    binary_var = ["open"]
    categorical_var = ["categories", "zip_clust"]


    col_names_train= train.columns.values
    col_names_pred = pred.columns.values

    numerical_var = [col for col in col_names_train if col not in binary_var if col not in categorical_var if col not in target_var]
    numerical_var_pred = [col for col in col_names_pred if col not in binary_var if col not in categorical_var]
    # encoded_categorical, encoded_binary = label_encoder(train, binary_var, categorical_var)
    # encoded_categorical_pred, encoded_binary_pred= label_encoder(pred, binary_var, categorical_var)
    encoded_binary = label_encoder(train, binary_var)
    encoded_binary_pred= label_encoder(pred, binary_var)

    categorical_variables = np.array(train[categorical_var])
    categorical_variables_pred = np.array(pred[categorical_var])
    numerical_data = np.array(train[numerical_var])
    numerical_data_pred = np.array(pred[numerical_var_pred])


    training_data_transformed = np.concatenate((categorical_variables,encoded_binary,numerical_data),axis=1)
    pred_data_transformed = np.concatenate((categorical_variables_pred,encoded_binary_pred,numerical_data_pred),axis=1)



    #train-test split
    # train_x = training_data_transformed[0:160378,]
    # train_y = target[0:160378]
    # test_x = training_data_transformed[160379:200473,]
    # test_y = target[160379:200473]
    # pred_x = pred_data_transformed

    train_x = training_data_transformed
    train_y = target
    # test_x = training_data_transformed[160379:200473,]
    # test_y = target[160379:200473]
    pred_x = pred_data_transformed

    #pd.DataFrame(pred_x).to_csv("test_labelenc.csv")

    #REMEMBER TO ENCODE THE CATEGORICAL VARS
    # train_x, test_x = dummy_encoder(train_x, pred_x, categorical_variable_list = list(range(0,2,1)))

    #normalizing (if required)
    # train_x_norm = normalize(train_x)
    #test_x_norm = normalize(test_x)
    # pred_x_norm = normalize(pred_x)


    # print train_x, train_y
    # return train_x, train_y,test_x, test_y, train_x_norm, test_x_norm, pred_x, pred_x_norm, bus_id
    # train_x, pred_x = dummy_encoder(train_x, pred_x,)
    # return train_x, train_y,train_x_norm, pred_x, pred_x_norm, bus_id



train_test()