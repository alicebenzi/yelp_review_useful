__author__ = 'alicebenziger'

from data_encoding import train_test
from sklearn.svm import SVR
from sklearn.cross_validation import cross_val_score


def support_vector_regressor(X_train, y_train):
    clf = SVR()
    clf.fit(X_train, y_train)
    # plot_learning_curve(clf, title ="Support Vector learning curve", X = X_train,y = y_train, ylim=(0, 1.1))
    y_pred_SVC = clf.predict(X_train)
    # plot_validation_curve(clf,title ="Support Vector Classifier validation curve", X=X_train, y=y_train,param_name='C', param_range = [1, 5, 20, 50])
    print y_train, y_pred_SVC
    print clf.score(X_train, y_train)
    # cross_validation_accuracy= cross_val_score(clf, X_train, y_train, cv = 5, scoring = 'accuracy').mean()
    # print "Support Vector Classifier : Training set metrics"
    # print "Cross validation accuracy:", cross_validation_accuracy
    # print_metrics(y_train, y_pred_SVC)
    # file = open("SVM_clf.p", "wb")
    # pickle.dump(clf, file)
    # file.close()

if __name__ == '__main__':

    train_x, train_y,test_x, test_y, train_x_norm, test_x_norm, pred_x = train_test()
    support_vector_regressor(train_x_norm,train_y)

