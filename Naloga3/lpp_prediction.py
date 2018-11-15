from Naloga3.linear import LinearLearner
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import numpy as np
import time
import psutil
import os

from Naloga3.lpputils import get_datetime, tsdiff

def limit_cpu():
    p = psutil.Process(os.getpid())
    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)

class LppPrediction(object):

    holidays = ["1.1", "2.1", "8.2", "27.4", "1.5", "2.5", "25.6", "15.8", "31.10", "1.11", "25.12", "26.12", "31.12"]

    seasons = ["winter", "spring", "summer", "fall"]
    onehot_encoded_features = ["dep_route"]
    #"hour_avg", "driver_dev_from_avg"
    y_dependent_features = []
    y_independent_features = ["dep_hour_sin", "dep_hour_cos", "dep_minute_sin", "dep_minute_cos", "weekends_holidays",
                "dep_month_sin", "dep_month_cos", "rush_hour"]
    features = y_dependent_features + y_dependent_features

    def __init__(self,data):
        self.data = data
        self.lines = list(set(self.data["Route"]))
        self.classifiers = dict.fromkeys(self.lines)
        self.line_features = dict.fromkeys(self.lines)
        self.trained = False
        self.trained_avg_hour = []
        self.trained_driver_dev_from_avg = dict()


    def __call__(self, X,line_idx):
        classifier = self.classifiers[line_idx]
        return classifier(X)

    def create_classifiers(self):

        print("Started creating classifiers...")
        start_time = time.time()

        lines_data = [self.data[self.data["Route"] == line].reset_index(drop=True) for line in self.lines]

        # Spread the workload among the cores of processors to speed up processing
        with ProcessPoolExecutor(max_workers=4) as executor:
            for classifier,line_features,line in executor.map(self.create_single_classifier,lines_data,self.lines):
                self.classifiers[line] = classifier
                self.line_features[line] = line_features

        self.trained = True

        print("Finished creating classifiers in %dm %ss." % ((time.time() - start_time) // 60,round(time.time() - start_time) % 60))

    def create_single_classifier(self,line_data,line):
        limit_cpu()
        train_df, y = self.preprocess_train_data(line_data)
        model = LineLearner()
        classifier = model(train_df.values, y)
        line_features = list(train_df.keys())
        print("Created classifier %d of %d." % (line,len(self.lines)))
        return classifier,line_features,line


    def preprocess_independent_data(self,data):
        dataframe = pd.DataFrame(columns=self.y_independent_features)



        # # FEATURE = departure hour
        if "dep_hour_sin" in self.y_independent_features:
            dataframe["dep_hour_sin"] = np.array(
                list(map(lambda x: np.sin(get_datetime(x).hour * (2.0 * np.pi / 24)), data["Departure time"])))

        if "dep_hour_cos" in self.y_independent_features:
            dataframe["dep_hour_cos"] = np.array(
               list(map(lambda x: np.cos(get_datetime(x).hour * (2.0 * np.pi / 24)), data["Departure time"])))

        # dataframe["dep_minute"] = np.array(list(map(lambda x: 0.0 if get_datetime(x).minute < 30 else 1.0,data["Departure time"])))
        if "dep_minute_sin" in self.y_independent_features:
            dataframe["dep_minute_sin"] = np.array(
                list(map(lambda x: np.sin(get_datetime(x).minute * (2.0 * np.pi / 60)), data["Departure time"])))

        if "dep_minute_cos" in self.y_independent_features:
            dataframe["dep_minute_cos"] = np.array(
                list(map(lambda x: np.cos(get_datetime(x).minute * (2.0 * np.pi / 60)), data["Departure time"])))

        # dataframe["dep_month"] = np.array(list(map(lambda x: 1.0 if get_datetime(x).month in [11,12] else 0.0,data["Departure time"])))
        if "dep_month_sin" in self.y_independent_features:
            dataframe["dep_month_sin"] = np.array(
                list(map(lambda x: np.sin(get_datetime(x).month * (2.0 * np.pi / 12)), data["Departure time"])))

        if "dep_month_cos" in self.y_independent_features:
            dataframe["dep_month_cos"] = np.array(
                list(map(lambda x: np.cos(get_datetime(x).month * (2.0 * np.pi / 12)), data["Departure time"])))

        if "winter" in self.y_independent_features:
            dataframe["winter"] = np.array(
                list(map(lambda x: 1 if get_datetime(x).month in [11, 12, 1, 2] else 0, data["Departure time"])))

        if "spring" in self.y_independent_features:
            dataframe["spring"] = np.array(
                list(map(lambda x: 1 if get_datetime(x).month in [3, 4, 5] else 0, data["Departure time"])))

        if "summer" in self.y_independent_features:
            dataframe["summer"] = np.array(
                list(map(lambda x: 1 if get_datetime(x).month in [6, 7, 8] else 0, data["Departure time"])))

        if "fall" in self.y_independent_features:
            dataframe["fall"] = np.array(
                list(map(lambda x: 1 if get_datetime(x).month in [9, 10] else 0, data["Departure time"])))

        if "rush_hour" in self.y_independent_features:
            dataframe["rush_hour"] = np.array(
                list(map(lambda x: 1 if get_datetime(x).hour in [6, 7, 8, 15, 16, 17] else 0, data["Departure time"])))

        if "weekends_holidays" in self.y_independent_features:
            dataframe["weekends_holidays"] = np.array(list(map(lambda x: 1 if (get_datetime(x).weekday() >= 5 or (
                    "%d.%d" % (get_datetime(x).day, get_datetime(x).month)) in self.holidays) else 0,
                                                               data["Departure time"])))
        return dataframe

    def preprocess_test_data(self,data):
        # Check whether we have y_dependent training data
        if(not self.trained):
            print("No training has been done for these models!")
            exit(-1)

        dataframe = self.preprocess_independent_data(data)

        half_hour_idx = 23

        if "hour_avg" in self.y_dependent_features:
            dataframe["hour_avg"] = np.array(list(map(
                lambda x: self.trained_avg_hour[get_datetime(x).hour] if get_datetime(x).minute < 30 else
                self.trained_avg_hour[get_datetime(x).hour + half_hour_idx], data["Departure time"])))

        if "driver_dev_from_avg" in self.y_dependent_features:
            dataframe["driver_dev_from_avg"] = np.array(list(
                map(lambda x: self.trained_driver_dev_from_avg[str(x)] if str(x) in self.trained_driver_dev_from_avg.keys() else 0.0,
                    data["Driver ID"])))

        if "dep_route" in self.onehot_encoded_features:
            df_dep_route = pd.get_dummies(data["Route Direction"])
            dataframe = pd.concat([dataframe,df_dep_route],axis=1,sort=False)

        return dataframe

    def preprocess_train_data(self,data):
        dataframe = self.preprocess_independent_data(data)

        # Calculate average drive time
        drive_times = [abs(tsdiff(get_datetime(row[0]), get_datetime(row[1]))) for row in
                       data[["Departure time", "Arrival time"]].values]
        avg_drive_time = sum(drive_times) / len(drive_times)
        # Calculate 30 minute average drive times
        half_hour_idx = 23
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
                tdiff = tsdiff(get_datetime(row[1]), get_datetime(row[0]))
                sum_time[hour + half_hour_idx] += tdiff
                cnt_time[hour + half_hour_idx] += 1
        avg_time = list(map(lambda sum, cnt: avg_drive_time if cnt == 0 else sum / cnt, sum_time, cnt_time))
        self.trained_avg_hour = avg_time

        # dataframe["target_var"] = np.array(list(map(lambda x: avg_time[get_datetime(x).hour] if get_datetime(x).minute < 30 else avg_time[get_datetime(x).hour+24], data["Departure time"])))
        if "hour_avg" in self.y_dependent_features:
            dataframe["hour_avg"] = np.array(list(map(
                lambda x: avg_time[get_datetime(x).hour] if get_datetime(x).minute < 30 else avg_time[
                    get_datetime(x).hour + half_hour_idx], data["Departure time"])))

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
            # print("Driver %d has an average error of %d" % (id,diff))

            if diff < -thresh:
                fast_drivers.append(id)
            elif diff > thresh:
                slow_drivers.append(id)
        self.trained_driver_dev_from_avg = driver_diff
        if "hour_avg" in self.y_dependent_features:
            dataframe["hour_avg"] = np.array(list(map(lambda x: driver_diff[str(x)], data["Driver ID"])))

        if "dep_route" in self.onehot_encoded_features:
            df_dep_route = pd.get_dummies(data["Route Direction"])
            dataframe = pd.concat([dataframe,df_dep_route],axis=1,sort=False)

        y = np.array(list(
            map(lambda x, y: abs(tsdiff(get_datetime(x), get_datetime(y))), data["Departure time"],
                data["Arrival time"])))

        self.trained = True
        return dataframe,y


class LineLearner(object):

    def __call__(self,X,Y,_lambda=1.0):
        model = LinearLearner(lambda_=_lambda)
        return model(X,Y)
