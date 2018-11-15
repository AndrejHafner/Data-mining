from datetime import datetime,timedelta

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%f"

def parsedate(x):
    if not isinstance(x, datetime):
        x = datetime.datetime.strptime(x, DATETIME_FORMAT)
    return x

def tsdiff(x, y):
    return (parsedate(x) - parsedate(y)).total_seconds()

def tsadd(x, seconds):
    d = timedelta(seconds=seconds)
    nd = parsedate(x) + d
    return nd.strftime(DATETIME_FORMAT)

def get_datetime(time_str):
    return datetime.strptime(time_str.replace(".000",""),DATETIME_FORMAT)

def get_day_seconds(time_str):
    dt = get_datetime(time_str)
    return dt.hour * 3600 + dt.minute * 60 + dt.second

if __name__ == "__main__":
    testd1 = "2012-01-01 23:32:38.000"
    testd2 = "2012-12-01 03:33:38.000"
    
    testd1 = datetime.strptime(testd1, DATETIME_FORMAT)

    for i in range(23000):
        a = tsdiff(testd1, testd2)
        b = tsadd(testd1, -122)
