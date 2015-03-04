__author__ = 'alicebenziger'
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVR
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

def count_vectoriser(text_col):
    vect = CountVectorizer(tokenizer=lambda text: text.split(','))
    cat_fea = vect.fit_transform(text_col.fillna(''))
    cat_fea = cat_fea.todense()
    idx_max_1 = cat_fea > 1
    cat_fea[idx_max_1] = 1

    return cat_fea




def normalize(X):
    normalizer = preprocessing.Normalizer().fit(X)
    normalized_X = normalizer.transform(X)
    return normalized_X



def train_test():

    train = pd.read_csv('train_new.csv', header=0)
    pred = pd.read_csv('test_new.csv', header=0)

    # print train.columns.values

    del train["business_id"], train["date"], train["review_id"], train["text"], train["type_x"], train["user_id"],\
        train["votes_cool_x"], train["votes_funny_x"],train["full_address"], train["latitude"], train["longitude"],\
        train["name_x"], train["neighborhoods"], train["type_y"], train["name_y"], train["type"], train["votes_cool_y"],\
        train["votes_funny_y"], train["votes_useful_y"], train["state"]

    #print test.columns.values
    del pred["business_id"], pred["date"], pred["review_id"], pred["text"], pred["type_x"], pred["user_id"],\
        pred["full_address"], pred["latitude"], pred["longitude"],\
        pred["name_x"], pred["neighborhoods"], pred["type_y"], pred["name_y"], pred["type"], pred["state"]

    col_names_train= train.columns.values
    col_names_pred = pred.columns.values

    #print col_names_train

    target = np.array(train["votes_useful_x"])
    target_var = ["votes_useful_x"]
    binary_var = ["open"]
    categorical_var = ["categories", "city", "zip_code"]

    #categorical_var = ["city"]

    category_encoded = count_vectoriser(train["categories"])
    zip_encoded = count_vectoriser(train["zip_code"])
    city_encoded = count_vectoriser(train["city"])
    #open_encoded = count_vectoriser(train["open"])
    categorical_variables = np.concatenate((category_encoded,zip_encoded,city_encoded),axis=1)

    category_encoded_pred = count_vectoriser(pred["categories"])
    zip_encoded_pred = count_vectoriser(pred["zip_code"])
    city_encoded_pred = count_vectoriser(pred["city"])
    #open_encoded = count_vectoriser(train["open"])
    categorical_variables = np.concatenate((category_encoded,zip_encoded,city_encoded),axis=1)
    categorical_variables_pred = np.concatenate((category_encoded_pred,zip_encoded_pred,city_encoded_pred),axis=1)

    # print category_encoded
    # print zip_encoded
    #print categorical_variables
    numerical_var = [col for col in col_names_train if col not in binary_var if col not in categorical_var if col not in target_var]
    numerical_var_pred = [col for col in col_names_pred if col not in binary_var if col not in categorical_var]
    # encoded_categorical, encoded_binary = label_encoder(train, binary_var, categorical_var)
    # encoded_categorical_pred, encoded_binary_pred= label_encoder(pred, binary_var, categorical_var)
    encoded_binary = label_encoder(train, binary_var)
    encoded_binary_pred= label_encoder(pred, binary_var)

    numerical_data = np.array(train[numerical_var])
    numerical_data_pred = np.array(pred[numerical_var_pred])


    training_data_transformed = np.concatenate((categorical_variables,encoded_binary,numerical_data),axis=1)
    pred_data_transformed = np.concatenate((categorical_variables_pred,encoded_binary_pred,numerical_data_pred),axis=1)



    #train-test split
    train_x = training_data_transformed[0:160378,]
    train_y = target[0:160378]
    test_x = training_data_transformed[160379:200473,]
    test_y = target[160379:200473]
    pred_x = pred_data_transformed

    pd.DataFrame(test_x).to_csv("test_labelenc.csv")
    train_x, test_x = dummy_encoder(train_x, test_x, categorical_variable_list = list(range(0,2,1)))

    #normalizing (if required)
    train_x_norm = normalize(train_x)
    test_x_norm = normalize(test_x)

    # print train_x, train_y
    return train_x, train_y,test_x, test_y, train_x_norm, test_x_norm, pred_x




    #quicktest
    # clf = SVR(C=1.0, epsilon=0.2)
    # clf.fit(train_x,train_y)
    # print clf.score(test_x, test_y)
train_x, train_y,test_x, test_y, train_x_norm, test_x_norm, pred_x = train_test()
