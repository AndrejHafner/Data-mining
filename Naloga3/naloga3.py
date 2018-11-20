import numpy as np
import pandas as pd
import math

from datetime import datetime,timedelta
from Naloga3.lpp_prediction import LppPrediction
from Naloga3.lpputils import get_datetime, tsdiff



def kfold(df,i,k = 10):
    n = len(df)
    indexes_to_keep = set(range(df.shape[0])) - set(range(n*(i-1)//k,(n*i//k)))
    train_df = df.take(list(indexes_to_keep))
    test_df = df.take(range(n*(i-1)//k,n*i//k))
    return train_df,test_df

def crossvalidate(data):

    errors = []


    print("Starting crossvalidation")
    for i in range(1,11):
        data_train_df, data_test_df = kfold(data,i)
        data_train_df = data_train_df.reset_index(drop=True)
        data_test_df = data_test_df.reset_index(drop=True)

        _sum = 0
        print("Iteration %d of 10" % i)

        predictor = LppPrediction(data_train_df)
        predictor.create_classifiers()
        testing_data = predictor.preprocess_test_data(data_test_df)
        testing_features = list(testing_data.keys())
        for row, line_idx, dep_time,k in zip(testing_data.values, data_test_df["Route"].values,data_test_df["Departure time"].values,range(len(data_test_df))):
            line_features = predictor.line_features[line_idx]
            row_features_idx = [i for i, e in enumerate(testing_features) if e in line_features]
            line_features_idx = [i for i, e in enumerate(line_features) if e in testing_features]

            entry = np.zeros(len(line_features))
            for i in range(len(line_features)):
                if i in line_features_idx:
                    entry[i] = row[row_features_idx[0]]
                    del row_features_idx[0]
                else:
                    entry[i] = 0.0
            pred_sec = predictor(entry, line_idx)

            pred_date = get_datetime(dep_time) + timedelta(seconds=pred_sec)

            actual_date = get_datetime(data_test_df["Arrival time"].values[k])

            _sum += abs(tsdiff(pred_date, actual_date))
        errors.append(_sum / len(data_test_df))

    return sum(errors) / len(errors)

def get_precipitation_for_date(weather_data,date):
    dt = get_datetime(date)
    row = weather_data[weather_data["date"] == dt.strftime("%d.%m.%Y")]
    return math.nan if len(row) == 0 else row["precipitation"].values.ravel()[0]




if __name__ == "__main__":
    train_data = pd.read_csv("data/train.csv", sep='\t')# FOR CROSSVALIDATION .sample(frac=0.1, random_state=42).reset_index(drop=True)
    test_data = pd.read_csv("data/test.csv",sep='\t')#.sample(frac=0.01, random_state=42).reset_index(drop=True)
    weather_data = pd.read_csv("data/weather_data.csv",sep=';')


    train_data = train_data[train_data.apply(lambda x: get_datetime(x["Departure time"]).month in [1,2,11],axis=1)].reset_index(drop=True)

    # Add weather data
    train_data["precipitation"] = np.array(
                list(map(lambda x: get_precipitation_for_date(weather_data,x), train_data["Departure time"])))
    test_data["precipitation"] = np.array(
        list(map(lambda x: get_precipitation_for_date(weather_data, x), test_data["Departure time"])))

    # CROSS VALIDATION
    # print(crossvalidate(train_data))

    # Prediction for test data (pipe for output to txt file)
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
        print(get_datetime(dep_time) +  timedelta(seconds = predictor(entry,line_idx)))

