import csv
import numpy as np
import pandas as pd
import time

from datetime import datetime,timedelta
from Naloga3.linear import LinearLearner, LinearRegClassifier

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%f"

# features = ["registration","driver_id","dep_month","dep_day","dep_hour","dep_minute","dep_second"]
days_features = ["day_%d" % (i+1) for i in range(31)]
features = ["dep_hour"]# + days_features

# Created by the lab assistant - Andrej Čopar
def parsedate(x):
    if not isinstance(x, datetime):
        x = datetime.strptime(x, DATETIME_FORMAT)
    return x

# Created by the lab assistant - Andrej Čopar
def tsdiff(x, y):
    return (parsedate(x) - parsedate(y)).total_seconds()

# Created by the lab assistant - Andrej Čopar
def tsadd(x, seconds):
    d = timedelta(seconds=seconds)
    nd = parsedate(x) + d
    return nd.strftime(DATETIME_FORMAT)

def get_datetime(time_str):
    return datetime.strptime(time_str.replace(".000",""),DATETIME_FORMAT)

def get_day_seconds(time_str):
    dt = get_datetime(time_str)
    return dt.hour * 3600 + dt.minute * 60 + dt.second

def preprocess_data(data, train = True):
    # Initialize the output data frame
    dataframe = pd.DataFrame(columns= features + ["target_var"])

    sum_time = [0 for i in range(0,23)]
    cnt_time = [0 for i in range(0,23)]
    if train:
        for row in data[["Departure time","Arrival time"]].values:
            hour = get_datetime(row[0]).hour
            sum_time[hour] += tsdiff(get_datetime(row[1]),get_datetime(row[0]))
            cnt_time[hour] += 1


    avg_time = list(map(lambda sum, cnt:30*60 if cnt == 0 else sum / cnt,sum_time,cnt_time))


    # FEATURE = driver ID -> the arrival time can depend on the driver
    # dataframe["driver_id"] = data["Driver ID"]
    #
    # # FEATURE = registration -> different vehicles can have different properties, thus taking different times
    # unique_registrations = list(set(data["Registration"]))
    # dataframe["registration"] = list(map(lambda x: unique_registrations.index(x),data["Registration"]))
    #
    # # FEATURE = departure month
    # dataframe["dep_month"] = np.array(list(map(lambda x: get_datetime(x).month,data["Departure time"])))
    #
    # # FEATURE = departure day -> can depend if its a weekened or a holiday
    # dataframe["dep_day"] = np.array(list(map(lambda x: get_datetime(x).day,data["Departure time"])))
    #
    # # FEATURE = departure hour
    dataframe["dep_hour"] = np.array(list(map(lambda x: get_datetime(x).hour,data["Departure time"])))
    #
    # # FEATURE = departure minute
    # dataframe["dep_minute"] = np.array(list(map(lambda x: get_datetime(x).minute, data["Departure time"])))
    #
    # # FEATURE = departure second
    # dataframe["dep_second"] = np.array(list(map(lambda x: get_datetime(x).second, data["Departure time"])))



    #dataframe["dep_time"] = np.array(list(map(lambda dep_time,arr_time: tsdiff(get_datetime(arr_time),get_datetime(dep_time)), data["Departure time"],data["Arrival time"])))

    if "weekend" in features:
        dataframe["weekend"] = np.array(list(map(lambda x: 1.0 if get_datetime(x).weekday() >= 5 else 0.0, data["Departure time"])))

    # Set for separate days
    if "day_1" in dataframe:
        for i in range(len(data)):
            day_idx = "day_" + str(get_datetime(data.get_value(i,"Departure time")).day)
            empty = days_features.copy()
            empty.remove(day_idx)
            dataframe.loc[i,empty] = float(0.0)
            dataframe.loc[i,day_idx] = float(0.0)
            if( i % 100 == 0):
                print("Created features for %d rows out of %d" % (i, len(data)))


    # Add the target variable to the dataframe
    if data["Arrival time"][1] != "?":
        dataframe["target_var"] = np.array(list(map(lambda x: avg_time[get_datetime(x).hour],data["Departure time"])))
        #dataframe["target_var"] = np.array(list(map(lambda x: round(time.mktime(get_datetime(x).timetuple())),data["Arrival time"])))
        # dataframe["target_var"] = np.array(list(map(lambda dep_time, arr_time: tsdiff(get_datetime(arr_time), get_datetime(dep_time)),
        #                   data["Departure time"], data["Arrival time"])))

    return dataframe

def calculate_error(mod_df, test_df):
    pass

def kfold(df,i,k = 10):
    n = len(df)
    indexes_to_keep = set(range(df.shape[0])) - set(range(n*(i-1)//k,(n*i//k)))
    train_df = df.take(list(indexes_to_keep))
    test_df = df.take(range(n*(i-1)//k,n*i//k))
    return train_df,test_df

def crossvalidate(df,data):

    errors = []
    model = LinearLearner(lambda_=0.0001)

    print("Starting crossvalidation")
    for i in range(1,11):
        train_df, test_df = kfold(df,i)
        data_train_df, data_test_df = kfold(data,i)
        _sum = 0
        vals = train_df[features].values
        classifier = model(vals,train_df["target_var"])

        # Evaluate the error
        print("Iteration %d of 10" % i)
        for j in range(len(test_df)):
            actual_date = get_datetime(data_test_df["Arrival time"].values[j])
            pred_seconds = classifier(test_df[features].values[j])
            pred_date = get_datetime(data_test_df["Departure time"].values[j]) + timedelta(seconds=pred_seconds)

            _sum += abs(tsdiff(pred_date,actual_date))
        errors.append(_sum / len(test_df))

    return sum(errors) / len(errors)






if __name__ == "__main__":
    data = pd.read_csv("data/train_pred.csv", sep='\t')#.sample(frac=1, random_state=42).reset_index(drop=True)
    sum_time = [0 for i in range(0, 23)]
    cnt_time = [0 for i in range(0, 23)]

    for row in data[["Departure time", "Arrival time"]].values:
        hour = get_datetime(row[0]).hour
        sum_time[hour] += tsdiff(get_datetime(row[1]), get_datetime(row[0]))
        cnt_time[hour] += 1

    avg_time = list(map(lambda sum, cnt: 30 * 60 if cnt == 0 else sum / cnt, sum_time, cnt_time))

    # train_df = preprocess_data(data)
    test_data = pd.read_csv("data/test_pred.csv",sep='\t')

    for i in range(len(test_data)):
        dep_time = get_datetime(test_data["Departure time"].values[i])
        print(dep_time + timedelta(seconds=avg_time[dep_time.hour]))

    # test_df = preprocess_data(test_data,train=False)
    # model = LinearLearner()
    # classifier = model(train_df[features].values,train_df["target_var"])
    # #print(crossvalidate(train_df,data))
    # i = 0
    # for row in test_df[features].values:
    #     deltas = classifier(row)
    #     print(get_datetime(test_data["Departure time"].values[i]) + timedelta(seconds = deltas))
    #     i += 1
    # print(classifier(10))

