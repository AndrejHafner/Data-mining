import csv
import numpy as np
import pandas as pd
import time

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

from datetime import datetime,timedelta
from Naloga3.linear import LinearLearner, LinearRegClassifier
from Naloga3.lpp_prediction import LppPrediction
from Naloga3.lpputils import get_datetime, tsdiff

holidays = ["1.1","2.1","8.2","27.4","1.5","2.5","25.6","15.8","31.10","1.11","25.12","26.12","31.12"]

seasons = ["winter","spring","summer","fall"]

features = ["dep_hour_sin","dep_hour_cos","dep_minute_sin","dep_minute_cos","weekends_holidays","dep_month_sin","dep_month_cos","rush_hour","hour_avg","driver_dev_from_avg"] + seasons


trained_avg_hour = []
trained_driver_dev_from_avg = dict()








def preprocess_data(data, train = True,trained_avg_time = None, trained_driver_dev = None):
    # Initialize the output data frame
    dataframe = pd.DataFrame(columns= features + ["target_var"])





    # FEATURE = driver ID -> the arrival time can depend on the driver
    # dataframe["driver_id"] = data["Driver ID"]
    #
    # # FEATURE = registration -> different vehicles can have different properties, thus taking different times
    # unique_registrations = list(set(data["Registration"]))
    # dataframe["registration"] = list(map(lambda x: unique_registrations.index(x),data["Registration"]))


    # # FEATURE = departure hour
    dataframe["dep_hour_sin"] = np.array(list(map(lambda x: np.sin(get_datetime(x).hour * (2.0 * np.pi/24)),data["Departure time"])))
    dataframe["dep_hour_cos"] = np.array(list(map(lambda x: np.cos(get_datetime(x).hour * (2.0 * np.pi/24)),data["Departure time"])))
    #dataframe["dep_minute"] = np.array(list(map(lambda x: 0.0 if get_datetime(x).minute < 30 else 1.0,data["Departure time"])))
    dataframe["dep_minute_sin"] = np.array(list(map(lambda x:np.sin(get_datetime(x).minute * (2.0 * np.pi/60)),data["Departure time"])))
    dataframe["dep_minute_cos"] = np.array(list(map(lambda x:np.cos(get_datetime(x).minute * (2.0 * np.pi/60)),data["Departure time"])))

    # dataframe["dep_month"] = np.array(list(map(lambda x: 1.0 if get_datetime(x).month in [11,12] else 0.0,data["Departure time"])))
    dataframe["dep_month_sin"] = np.array(list(map(lambda x: np.sin(get_datetime(x).month * (2.0 * np.pi/12)),data["Departure time"])))
    dataframe["dep_month_cos"] = np.array(list(map(lambda x: np.cos(get_datetime(x).month * (2.0 * np.pi/12)),data["Departure time"])))

    dataframe["winter"] = np.array(list(map(lambda x: 1 if get_datetime(x).month in [11,12,1,2] else 0,data["Departure time"])))
    dataframe["spring"] = np.array(list(map(lambda x: 1 if get_datetime(x).month in [3,4,5] else 0,data["Departure time"])))
    dataframe["summer"] = np.array(list(map(lambda x: 1 if get_datetime(x).month in [6,7,8] else 0,data["Departure time"])))
    dataframe["fall"] = np.array(list(map(lambda x: 1 if get_datetime(x).month in [9,10] else 0,data["Departure time"])))


    dataframe["rush_hour"] = np.array(list(map(lambda x: 1 if get_datetime(x).hour in [6,7,8,15,16,17] else 0,data["Departure time"])))

    if "weekends_holidays" in features:
        dataframe["weekends_holidays"] = np.array(list(map(lambda x: 1 if (get_datetime(x).weekday() >= 5 or ("%d.%d" % (get_datetime(x).day,get_datetime(x).month)) in holidays) else 0, data["Departure time"])))




    if train:
        # Calculate average drive time
        drive_times = [abs(tsdiff(get_datetime(row[0]),get_datetime(row[1]))) for row in data[["Departure time", "Arrival time"]].values]
        avg_drive_time = sum(drive_times) / len(drive_times)
        # Calculate 30 minute average drive times
        sum_time = [0 for i in range(0, 47)]
        cnt_time = [0 for i in range(0, 47)]
        for row in data[["Departure time", "Arrival time"]].values:
            # if get_datetime(row[0]).month in [6, 7, 8]: # May, June, July, August, September
            #     continue
            hour = get_datetime(row[0]).hour
            minute = get_datetime(row[0]).minute
            if minute < 30:
                sum_time[hour] += tsdiff(get_datetime(row[1]), get_datetime(row[0]))
                cnt_time[hour] += 1
            else:
                sum_time[hour+24] += tsdiff(get_datetime(row[1]), get_datetime(row[0]))
                cnt_time[hour+24] += 1
        avg_time = list(map(lambda sum, cnt: avg_drive_time if cnt == 0 else sum / cnt, sum_time, cnt_time))
        trained_avg_hour = avg_time

        # dataframe["target_var"] = np.array(list(map(lambda x: avg_time[get_datetime(x).hour] if get_datetime(x).minute < 30 else avg_time[get_datetime(x).hour+24], data["Departure time"])))
        dataframe["target_var"] = np.array(list(map(lambda x,y: abs(tsdiff(get_datetime(x),get_datetime(y))), data["Departure time"],data["Arrival time"])))
        dataframe["hour_avg"] = np.array(list(map(lambda x: avg_time[get_datetime(x).hour] if get_datetime(x).minute < 30 else avg_time[get_datetime(x).hour+24],data["Departure time"])))

        # Find slow drivers
        driver_ids = set(data["Driver ID"])
        slow_drivers = []
        fast_drivers = []
        thresh = 90
        driver_diff = dict()
        for id in driver_ids:
            single_driver = data[data["Driver ID"] == id]
            _sum = 0
            cnt = 0
            for drive in single_driver[["Departure time", "Arrival time"]].values:
                _sum += abs(tsdiff(get_datetime(drive[0]), get_datetime(drive[1])))
                cnt += 1
            driver_avg = _sum / cnt
            diff = (driver_avg - avg_drive_time)
            driver_diff[str(id)] = diff
            #print("Driver %d has an average error of %d" % (id,diff))

            if diff < -thresh:
                fast_drivers.append(id)
            elif diff > thresh:
                slow_drivers.append(id)
        trained_driver_dev_from_avg = driver_diff
        dataframe["driver_dev_from_avg"] = np.array(list(map(lambda x: driver_diff[str(x)] , data["Driver ID"])))
        # dataframe["slow_driver"] = np.array(list(map(lambda x: 1 if x in slow_drivers else 0 , data["Driver ID"])))
        # dataframe["fast_driver"] = np.array(list(map(lambda x: 1 if x in fast_drivers else 0 , data["Driver ID"])))
    else:
        if trained_avg_time != None and trained_driver_dev != None:
            dataframe["hour_avg"] = np.array(list(map(lambda x: trained_avg_time[get_datetime(x).hour] if get_datetime(x).minute < 30 else trained_avg_time[get_datetime(x).hour+24],data["Departure time"])))
            dataframe["driver_dev_from_avg"] = np.array(list(map(lambda x: trained_driver_dev[str(x)] if str(x) in trained_driver_dev.keys() else 0.0 , data["Driver ID"])))
            trained_avg_hour, trained_driver_dev_from_avg = [],[]









    #dataframe["dep_time"] = np.array(list(map(lambda dep_time,arr_time: tsdiff(get_datetime(arr_time),get_datetime(dep_time)), data["Departure time"],data["Arrival time"])))




    # Add the target variable to the dataframe
    # if data["Arrival time"][1] != "?":
    #     dataframe["target_var"] = np.array(list(map(lambda x: avg_time[get_datetime(x).hour],data["Departure time"])))
        #dataframe["target_var"] = np.array(list(map(lambda x: round(time.mktime(get_datetime(x).timetuple())),data["Arrival time"])))
        # dataframe["target_var"] = np.array(list(map(lambda dep_time, arr_time: tsdiff(get_datetime(arr_time), get_datetime(dep_time)),
        #                   data["Departure time"], data["Arrival time"])))
    return dataframe,trained_avg_hour ,trained_driver_dev_from_avg


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
        vals = train_df[features].values
        # classifier = model(vals,train_df["target_var"])
        # print("Model thetas: " + str(list(classifier.th)))
        #classifier.th[2] = classifier.th[3] * 2
        poly = PolynomialFeatures(degree=2)
        X_ = poly.fit_transform(vals)
        predict_ = poly.fit_transform(test_df[features].values)

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
    test_data = pd.read_csv("data/test.csv",sep='\t')#.sample(frac=0.001, random_state=42).reset_index(drop=True)


    train_data = train_data[train_data.apply(lambda x: get_datetime(x["Departure time"]).month in [1,10,11],axis=1)].reset_index(drop=True)
    # line14 = train_data[train_data["Route"] == 14]
    predictor = LppPrediction(train_data)
    trained_data = predictor.create_classifiers()
    testing_data = predictor.preprocess_test_data(test_data)
    testing_features = list(testing_data.keys())
    for row,line_idx,dep_time in zip(testing_data.values,test_data["Route"].values,test_data["Departure time"].values):
        line_features = set(predictor.line_features[line_idx])
        intersection_features = line_features & set(testing_features)
        line_features_idx = [i for i, e in enumerate(testing_features) if e in line_features]
        zeros_to_add = abs(len(line_features - intersection_features))
        entry = row[line_features_idx]
        entry = np.ravel(entry)
        entry = np.pad(entry,(0,zeros_to_add),"constant")
        print(get_datetime(dep_time) +  timedelta(seconds = predictor(entry,line_idx)))
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

