from Naloga3.linear import LinearLearner
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import numpy as np
import time
import psutil
import os
import calendar
from Naloga3.lpputils import get_datetime, tsdiff

def limit_cpu():
    p = psutil.Process(os.getpid())
    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)

class LppPrediction(object):

    holidays = ["1.1", "2.1", "8.2", "27.4", "1.5", "2.5", "25.6", "15.8","29.10","30.10", "31.10", "1.11","2.11","21.12","24,12", "25.12","31.12"]

    onehot_encoded_features = ["dep_station","arr_station","precipitation","driver","hours","dep_route","minutes"] # 166.6 without precip
    #"hour_avg", "driver_dev_from_avg","days","minutes","days","months","dep_route",
    y_dependent_features = []
    y_independent_features = ["weekends","holidays", "rush_hour","dep_day_sin","dep_day_cos" ,"dep_month_sin", "dep_month_cos"] #+seasons
    # ,"dep_day_sin","dep_day_cos",,"dep_minute_sin","dep_minute_cos"
    #                 "dep_month_sin", "dep_month_cos",
    # ,"dep_hour_sin", "dep_hour_cos",
    features = y_dependent_features + y_independent_features

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

        # dataframe["dep_minute"] = np.array(list(map(lambda x: 0.0 if get_datetime(x).minute < 30 else 1.0,data["Departure time"])))
        if "dep_minute_sin" in self.y_independent_features:
            dataframe["dep_minute_sin"] = np.array(
                list(map(lambda x: np.sin(get_datetime(x).minute * (2.0 * np.pi / 60)), data["Departure time"])))

        if "dep_minute_cos" in self.y_independent_features:
            dataframe["dep_minute_cos"] = np.array(
                list(map(lambda x: np.cos(get_datetime(x).minute * (2.0 * np.pi / 60)), data["Departure time"])))
        # # FEATURE = departure hour
        if "dep_hour_sin" in self.y_independent_features:
            dataframe["dep_hour_sin"] = np.array(
                list(map(lambda x: np.sin(get_datetime(x).hour * (2.0 * np.pi / 24)), data["Departure time"])))

        if "dep_hour_cos" in self.y_independent_features:
            dataframe["dep_hour_cos"] = np.array(
               list(map(lambda x: np.cos(get_datetime(x).hour * (2.0 * np.pi / 24)), data["Departure time"])))

        if "dep_day_sin" in self.y_independent_features:
            dataframe["dep_day_sin"] = np.array(
                list(map(lambda x: np.sin(get_datetime(x).day * (2.0 * np.pi / calendar.monthrange(2012, get_datetime(x).month)[1])), data["Departure time"])))

        if "dep_day_cos" in self.y_independent_features:
            dataframe["dep_day_cos"] = np.array(
               list(map(lambda x: np.cos(get_datetime(x).day * (2.0 * np.pi / calendar.monthrange(2012, get_datetime(x).month)[1])), data["Departure time"])))

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
                list(map(lambda x: 1 if get_datetime(x).hour in [6, 7, 8, 15, 16, 17] and get_datetime(x).weekday() < 5 else 0, data["Departure time"])))

        if "weekends" in self.y_independent_features:
            dataframe["weekends"] = np.array(list(map(lambda x: 1 if get_datetime(x).weekday() >= 5 else 0,
                                                               data["Departure time"])))

        if "holidays" in self.y_independent_features:
            dataframe["holidays"] = np.array(list(map(lambda x: 1 if ("%d.%d" % (get_datetime(x).day, get_datetime(x).month) in self.holidays) else 0,
                                                      data["Departure time"])))

        if "precipitation" in self.onehot_encoded_features:
            data["precipitation"] = pd.Categorical(data["precipitation"],categories=["rain","snow"])
            df_precipitation = pd.get_dummies(data["precipitation"])
            dataframe = pd.concat([dataframe,df_precipitation],axis=1,sort=False)

        if "driver" in self.onehot_encoded_features:
            df_driver = pd.get_dummies(data["Driver ID"],prefix="driver_")
            dataframe = pd.concat([dataframe,df_driver],axis=1,sort=False)

        if "registration" in self.onehot_encoded_features:
            df_registration = pd.get_dummies(data["Registration"], prefix="reg_")
            dataframe = pd.concat([dataframe, df_registration], axis=1, sort=False)

        if "hours" in self.onehot_encoded_features:
            df_hours = pd.get_dummies(list(map(lambda x: get_datetime(x).hour , data["Departure time"])), prefix="hour_")
            dataframe = pd.concat([dataframe, df_hours], axis=1, sort=False)

        if "minutes" in self.onehot_encoded_features:
            df_minutes = pd.get_dummies(list(map(lambda x: (get_datetime(x).minute // 20) + 1 , data["Departure time"])), prefix="minutes_")
            dataframe = pd.concat([dataframe, df_minutes], axis=1, sort=False)

        if "days" in self.onehot_encoded_features:
            df_days = pd.get_dummies(list(map(lambda x: get_datetime(x).day , data["Departure time"])), prefix="day_")
            dataframe = pd.concat([dataframe, df_days], axis=1, sort=False)

        if "months" in self.onehot_encoded_features:
            df_month = pd.get_dummies(list(map(lambda x: get_datetime(x).month , data["Departure time"])), prefix="month_")
            dataframe = pd.concat([dataframe, df_month], axis=1, sort=False)


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

        if "dep_station" in self.onehot_encoded_features:
            df_dep_station = pd.get_dummies(data["First station"],prefix="dep_station_")
            dataframe = pd.concat([dataframe,df_dep_station],axis=1,sort=False)

        if "arr_station" in self.onehot_encoded_features:
            df_arr_station = pd.get_dummies(data["Last station"],prefix="arr_station_")
            dataframe = pd.concat([dataframe,df_arr_station],axis=1,sort=False)

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

        if "dep_station" in self.onehot_encoded_features:
            df_dep_station = pd.get_dummies(data["First station"],prefix="dep_station_")
            dataframe = pd.concat([dataframe,df_dep_station],axis=1,sort=False)

        if "arr_station" in self.onehot_encoded_features:
            df_arr_station = pd.get_dummies(data["Last station"],prefix="arr_station_")
            dataframe = pd.concat([dataframe,df_arr_station],axis=1,sort=False)

        if "dep_route" in self.onehot_encoded_features:
            df_dep_route = pd.get_dummies(data["Route Direction"])
            dataframe = pd.concat([dataframe,df_dep_route],axis=1,sort=False)



        y = np.array(list(
            map(lambda x, y: abs(tsdiff(get_datetime(x), get_datetime(y))), data["Departure time"],
                data["Arrival time"])))

        remove_idx = []
        for i in range(len(y)):
            if y[i] > 60*60*2:
                remove_idx.append(i)

        y = np.delete(y,remove_idx,0)
        dataframe = dataframe.drop(remove_idx).reset_index(drop=True)

        self.trained = True
        return dataframe,y


class LineLearner(object):

    def __call__(self,X,Y,_lambda=0.01):
        model = LinearLearner(lambda_=_lambda)
        return model(X,Y)
