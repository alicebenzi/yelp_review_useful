__author__ = 'alicebenziger'

from data_encoding import train_test
from sklearn.svm import SVR
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.cross_validation import cross_val_score

def rsmle_(predicted,actual):
   return np.sqrt(np.mean((pow(np.log(predicted+1) - np.log(actual+1),2))))

def support_vector_regressor(X_train, y_train):
    clf = SVR()
    clf.fit(X_train, y_train)
    pd.DataFrame(clf.predict(X_train)).to_csv("y_pred_svr.csv")
    print np.sqrt(np.mean((pow(np.log(clf.predict(X_train)+1) - np.log(y_train+1),2))))


if __name__ == '__main__':

    train_x, train_y, pred_x, train_x_norm, pred_x_norm, review_id = train_test()

    print "data fetched..."

    rf = RandomForestRegressor()
    rf.fit(train_x, train_y)

    print rf.score(train_x, train_y)
    Votes = rf.predict(pred_x)[:,np.newaxis]
    Id = np.array(review_id)[:,np.newaxis]
    print len(Votes), len(Id)


    submission_rf= np.concatenate((Id,Votes),axis=1)
    # print submission_rf
    np.savetxt("submission_rf.csv", submission_rf,header="Id,Votes", delimiter=',',fmt="%s, %0.2f", comments='')

    print "rf done"






    # gbr = GradientBoostingRegressor()
    # gbr.fit(train_x, train_y)
    # print gbr.score(train_x, train_y)
    # Votes = gbr.predict(pred_x)[:,np.newaxis]
    # Id = np.array(review_id)[:,np.newaxis]
    # print len(Votes), len(Id)
    #
    #
    # submission_gbr= np.concatenate((Id,Votes),axis=1)
    # # print submission_rf
    # np.savetxt("submission_gbr.csv", submission_gbr,header="Id,Votes", delimiter=',',fmt="%s, %0.2f", comments='')
