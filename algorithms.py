__author__ = 'alicebenziger'

from data_encoding import train_test
# from sklearn.svm import LinearSVR
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.learning_curve import validation_curve
from sklearn.learning_curve import learning_curve





def rsmle_(predicted,actual):
    print " i am called..."
    actual = np.exp(actual)-1
    predicted = np.exp(predicted)-1
    return np.sqrt(np.mean((pow(np.log(predicted+1) - np.log(actual+1),2))))

#customising the score for cross validation
rsmle_score = make_scorer(rsmle_,greater_is_better=False)

def plot_validation_curve(estimator, title, X, y, param_name, param_range, ylim=None):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel(param_name)
    plt.ylabel("Score")
    rsmle_score = make_scorer(rsmle_,greater_is_better=True)
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=5, scoring=rsmle_score, n_jobs=1)
    print "cross validation done...plotting the graph"
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.plot(param_range, train_scores_mean, label="Training score", color="r")
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2, color="r")
    plt.plot(param_range, test_scores_mean, label="Cross-validation score",
                 color="g")
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2, color="g")
    plt.legend(loc="best")
    plt.show()

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    rsmle_score = make_scorer(rsmle_,greater_is_better=True)

    train_sizes, train_scores, test_scores = learning_curve(
    estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,scoring = rsmle_score)
    print "plotting learning curve.."
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()


def random_forest_regressor(train_x, train_y, pred_x, review_id):
    # rf = RandomForestRegressor()
    rf = RandomForestRegressor(n_estimators=60,criterion='mse', max_depth=10)
    rf.fit(train_x, np.log(train_y+1))
    print rf.score(train_x, np.log(train_y+1))
    rf_pred = np.exp(rf.predict(pred_x))-1.0
    train_y = np.log(train_y+1.0)
    plot_validation_curve(rf, "Rf: Validation Curve", train_x,train_y,'min_samples_split',[2,3,4,5,10])
    # plot_learning_curve(rf,"Rf: Learning Curve", train_x,train_y)
    Votes = rf_pred[:,np.newaxis]
    Id = np.array(review_id)[:,np.newaxis]
    print len(Votes), len(Id)
    submission_rf= np.concatenate((Id,Votes),axis=1)
    # print submission_rf

    np.savetxt("submission_rf.csv", submission_rf,header="Id,Votes", delimiter=',',fmt="%s, %0.2f", comments='')


def ada_boost_regressor(train_x, train_y, pred_x, review_id):
    rng = np.random.RandomState(1)

    ada = AdaBoostRegressor(DecisionTreeRegressor(max_depth=1),n_estimators=300, random_state=rng)
    ada.fit(train_x, np.log(train_y+1))

    print ada.score(train_x, np.log(train_y+1))
    ada_pred = np.exp(ada.predict(pred_x))-1.0
    Votes = ada_pred[:,np.newaxis]
    Id = np.array(review_id)[:,np.newaxis]
    print len(Votes), len(Id)


    submission_ada= np.concatenate((Id,Votes),axis=1)
    # print submission_rf
    # np.savetxt("submission_ada.csv", submission_ada,header="Id,Votes", delimiter=',',fmt="%s, %0.2f", comments='')


if __name__ == '__main__':

    train_x, train_y, pred_x, train_x_norm, pred_x_norm, review_id = train_test()

    print "data fetched..."
    #random forest regressor
    random_forest_regressor(train_x, train_y, pred_x, review_id)
    print "done.."





