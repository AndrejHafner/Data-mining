import csv
import numpy as np
import pandas as pd
import time

from datetime import datetime,timedelta
from Naloga3.linear import LinearLearner, LinearRegClassifier

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

# features = ["registration","driver_id","dep_month","dep_day","dep_hour","dep_minute","dep_second"]
features = ["dep_time","weekend"]

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

def preprocess_data(data):
    # Initialize the output data frame
    dataframe = pd.DataFrame(columns= features + ["target_var"])

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
    # dataframe["dep_hour"] = np.array(list(map(lambda x: get_datetime(x).hour,data["Departure time"])))
    #
    # # FEATURE = departure minute
    # dataframe["dep_minute"] = np.array(list(map(lambda x: get_datetime(x).minute, data["Departure time"])))
    #
    # # FEATURE = departure second
    # dataframe["dep_second"] = np.array(list(map(lambda x: get_datetime(x).second, data["Departure time"])))
    dataframe["dep_time"] = np.array(list(map(lambda x: round(time.mktime(get_datetime(x).timetuple())), data["Departure time"])))
    dataframe["weekend"] = np.array(list(map(lambda x: 1 if get_datetime(x).weekday() >= 5 else 0, data["Departure time"])))


    # Add the target variable to the dataframe
    if data["Arrival time"][1] != "?":
        dataframe["target_var"] = np.array(list(map(lambda x: round(time.mktime(get_datetime(x).timetuple())),data["Arrival time"])))

    return dataframe

def calculate_error(mod_df, test_df):
    pass

def kfold(df,i,k = 10):
    n = len(df)
    indexes_to_keep = set(range(df.shape[0])) - set(range(n*(i-1)//k,(n*i//k)))
    train_df = df.take(list(indexes_to_keep))
    test_df = df.take(range(n*(i-1)//k,n*i//k))
    return train_df,test_df

def crossvalidate(df):

    errors = []
    model = LinearLearner()

    print("Starting crossvalidation")
    for i in range(1,11):
        train_df, test_df = kfold(df,i)
        _sum = 0
        classifier = model(train_df[features].values,train_df["target_var"])

        # Evaluate the error
        print("Iteration %d of 10" % i)
        for j in range(len(test_df)):
            pred_date = datetime.fromtimestamp(classifier(test_df[features].values[j]))
            actual_date = datetime.fromtimestamp(test_df["target_var"].values[j])
            _sum += abs(tsdiff(pred_date,actual_date))
        errors.append(_sum / len(test_df))

    return sum(errors) / len(errors)






if __name__ == "__main__":
    train_df = preprocess_data(pd.read_csv("data/train_pred.csv",sep='\t'))
    #test_df = preprocess_data(pd.read_csv("data/test_pred.csv",sep='\t'))
    # model = LinearLearner()
    # classifier = model(train_df[features].values,train_df["target_var"])
    print(crossvalidate(train_df))

    # for row in test_df[features].values:
    #     posix = classifier(row)
    #     print(datetime.fromtimestamp(posix).strftime(DATETIME_FORMAT)+".000")
    # print(classifier(10))

