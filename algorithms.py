__author__ = 'alicebenziger'

from data_encoding import train_test
from sklearn.svm import SVR
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from sklearn.cross_validation import cross_val_score

def rsmle_(predicted,actual):
   return np.sqrt(np.mean((pow(np.log(predicted+1) - np.log(actual+1),2))))

def support_vector_regressor(X_train, y_train):
    clf = SVR()
    clf.fit(X_train, y_train)
    # plot_learning_curve(clf, title ="Support Vector learning curve", X = X_train,y = y_train, ylim=(0, 1.1))
    #y_pred_SVC = clf.predict(X_train)
    #print clf.predict(X_train)
    pd.DataFrame(clf.predict(X_train)).to_csv("y_pred_svr.csv")
    print np.sqrt(np.mean((pow(np.log(clf.predict(X_train)+1) - np.log(y_train+1),2))))
    # plot_validation_curve(clf,title ="Support Vector Classifier validation curve", X=X_train, y=y_train,param_name='C', param_range = [1, 5, 20, 50])
    # print y_train, y_pred_SVC
    # print clf.score(X_train, y_train)
    # # cross_validation_accuracy= cross_val_score(clf, X_train, y_train, cv = 5, scoring = 'accuracy').mean()
    # print "Support Vector Classifier : Training set metrics"
    # print "Cross validation accuracy:", cross_validation_accuracy
    # print_metrics(y_train, y_pred_SVC)
    # file = open("SVM_clf.p", "wb")
    # pickle.dump(clf, file)
    # file.close()

if __name__ == '__main__':

    train_x, train_y,test_x, test_y, train_x_norm, test_x_norm, pred_x = train_test()


    rf = RandomForestRegressor()
    rf.fit(train_x, train_y)
    print rf.score(train_x,train_y)
    print rf.score(test_x,test_y)
    print rsmle_(rf.predict(test_x),test_y)
    #print train_x[1:5,], train_y[1:5], test_x[1:5,],test_y[1:5]
    # support_vector_regressor(train_x_norm,train_y)

