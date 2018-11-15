import csv
import numpy as np
import pandas as pd
import time

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

from datetime import datetime,timedelta
from Naloga3.lpp_prediction import LppPrediction
from Naloga3.lpputils import get_datetime, tsdiff



def kfold(df,i,k = 10):
    n = len(df)
    indexes_to_keep = set(range(df.shape[0])) - set(range(n*(i-1)//k,(n*i//k)))
    train_df = df.take(list(indexes_to_keep))
    test_df = df.take(range(n*(i-1)//k,n*i//k))
    return train_df,test_df

def crossvalidate(df,data):

    errors = []
    # model = LinearLearner(lambda_=0.1)

    poly = PolynomialFeatures(degree=2)

    print("Starting crossvalidation")
    for i in range(1,11):
        train_df, test_df = kfold(df,i)
        data_train_df, data_test_df = kfold(data,i)
        _sum = 0
        vals = train_df[LppPrediction.features].values
        # classifier = model(vals,train_df["target_var"])
        # print("Model thetas: " + str(list(classifier.th)))
        #classifier.th[2] = classifier.th[3] * 2
        poly = PolynomialFeatures(degree=2)
        X_ = poly.fit_transform(vals)
        predict_ = poly.fit_transform(test_df[LppPrediction.features].values)

        clf = linear_model.LinearRegression()
        clf.fit(X_, train_df["target_var"])
        pred_seconds = clf.predict(predict_)

        # Evaluate the error
        print("Iteration %d of 10" % i)
        for j in range(len(test_df)):
            actual_date = get_datetime(data_test_df["Arrival time"].values[j])
            # pred_seconds = classifier(test_df[features].values[j])
           # predict_ = poly.fit_transform(test_df[features].values[j])
            pred_date = get_datetime(data_test_df["Departure time"].values[j]) + timedelta(seconds=pred_seconds[j])

            _sum += abs(tsdiff(pred_date,actual_date))
        errors.append(_sum / len(test_df))

    return sum(errors) / len(errors)






if __name__ == "__main__":
    train_data = pd.read_csv("data/train.csv", sep='\t')#.sample(frac=0.01, random_state=42).reset_index(drop=True)
    test_data = pd.read_csv("data/test.csv",sep='\t')#".sample(frac=0.01, random_state=42).reset_index(drop=True)

    #TODO
    train_data = train_data[train_data.apply(lambda x: get_datetime(x["Departure time"]).month in [1,10,11],axis=1)].reset_index(drop=True)

    predictor = LppPrediction(train_data)
    trained_data = predictor.create_classifiers()
    testing_data = predictor.preprocess_test_data(test_data)
    testing_features = list(testing_data.keys())
    for row,line_idx,dep_time in zip(testing_data.values,test_data["Route"].values,test_data["Departure time"].values):
        line_features = predictor.line_features[line_idx]
        intersection_features = set(line_features) & set(testing_features)
        row_features_idx = [i for i, e in enumerate(testing_features) if e in line_features]
        line_features_idx = [i for i, e in enumerate(line_features) if e in testing_features]

        entry = np.zeros(len(line_features))
        for i in range(len(line_features)):
            if i in line_features_idx:
                entry[i] = row[row_features_idx[0]]
                del row_features_idx[0]
            else:
                entry[i] = 0.0

        #entry = np.array([row[i] if i in line_features_idx else -666 for i in range(len(line_features))])
        # zeros_to_add = abs(len(set(line_features) - intersection_features))
        # #FIXME pravilno dodanje ničel na prava mesta?
        # entry = row[line_features_idx]
        # entry = np.ravel(entry)
        # entry = np.pad(entry,(0,zeros_to_add),"constant")
        print(get_datetime(dep_time) +  timedelta(seconds = predictor(entry,line_idx)))

        # BEFORE
        # line_features = predictor.line_features[line_idx]
        # intersection_features = set(line_features) & set(testing_features)
        # line_features_idx = [i for i, e in enumerate(testing_features) if e in line_features]
        # missing_featurs_idx = [i for i, e in enumerate(predictor.line_features[line_idx]) if e not in testing_features]
        # zeros_to_add = abs(len(set(line_features) - intersection_features))
        # # FIXME pravilno dodanje ničel na prava mesta?
        # entry = row[line_features_idx]
        # entry = np.ravel(entry)
        # entry = np.pad(entry, (0, zeros_to_add), "constant")
        # print(get_datetime(dep_time) + timedelta(seconds=predictor(entry, line_idx)))

    # train_data = train_data[train_data["Route"] == 14]
    # train_df,trained_avg_hour,trained_driver_dev_from_avg = preprocess_data(train_data)
    #
    # # print(crossvalidate(train_df,train_data))
    #
    # #
    # #
    # test_df,not1,not2 = preprocess_data(test_data,train=False,trained_driver_dev = trained_driver_dev_from_avg, trained_avg_time = trained_avg_hour)
    # print("Starting to predict arrival times...")
    # vals = train_df[features].values
    # # classifier = model(vals,train_df["target_var"])
    # # print("Model thetas: " + str(list(classifier.th)))
    # # classifier.th[2] = classifier.th[3] * 2
    # poly = PolynomialFeatures(degree=2)
    # X_ = poly.fit_transform(vals)
    # predict_ = poly.fit_transform(test_df[features].values)
    #
    # clf = linear_model.LinearRegression()
    # clf.fit(X_, train_df["target_var"])
    # pred_seconds = clf.predict(predict_)
    #
    # i = 0
    # for row in range(len(pred_seconds)):
    #     deltas = pred_seconds[row]
    #     print(get_datetime(test_data["Departure time"].values[i]) + timedelta(seconds = deltas))
    #     i += 1


    #
    # test_df,not1,not2 = preprocess_data(test_data,train=False,trained_driver_dev = trained_driver_dev_from_avg, trained_avg_time = trained_avg_hour)
    # print("Starting to predict arrival times...")
    # model = LinearLearner(lambda_=1.0)
    # classifier = model(train_df[features].values,train_df["target_var"])
    #
    # i = 0
    # for row in test_df[features].values:
    #     deltas = classifier(row)
    #     print(get_datetime(test_data["Departure time"].values[i]) + timedelta(seconds = deltas))
    #     i += 1

