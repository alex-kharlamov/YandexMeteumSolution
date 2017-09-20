
import sys
from multiprocessing.pool import Pool

TRAIN_PATH = "./data/train_spb.tsv"
NETATMO_PATH = "./data/train_spb_netatmo.tsv"
TEST_PATH = "./data/test_spb_features.tsv"
TEST_NETATMO_PATH = "./data/test_spb_netatmo.tsv"

CITY_FEATURES_PATH = "./saved_data_spb"


import numpy as np
import pandas as pd
import json
import pickle

from tqdm import tqdm
from six.moves import zip as izip, map as imap


train = pd.read_csv(TRAIN_PATH, sep='\t',dtype=json.load(open("./data/train_col_dtypes.json")))


from sklearn.neighbors import KDTree
def preprocess_netatmo(df):
    """organizes netatmo stations into KDTrees for each distinct time frame"""
    df_by_hour = df.groupby('hour_hash')
    anns = {}
    for hour,stations_group in df_by_hour:
        anns[hour] = KDTree(stations_group[["netatmo_latitude","netatmo_longitude"]].values,metric='minkowski',p=2)
    
    #convert groupby to dict to get faster queries
    df_by_hour = {group:stations_group for group,stations_group in df_by_hour}
    return df_by_hour,anns
        

netatmo = pd.read_csv(NETATMO_PATH,na_values="None",sep='\t',dtype={'hour_hash':"uint64"})

netatmo_groups,netatmo_anns = preprocess_netatmo(netatmo)


def init_process(netatmo_groups, netatmo_anns):
    extract_features.netatmo_groups = netatmo_groups
    extract_features.netatmo_anns = netatmo_anns
    
def extract_features(group):
    """
    Extracts all kinds of features from a dataframe containing users in one group
    """
    netatmo_groups, netatmo_anns = extract_features.netatmo_groups, extract_features.netatmo_anns
    features = {}

    #square features
    square = {col: group[col].iloc[0] for col in group.columns}
    
    features['square_lat'] = square['sq_lat']
    features['square_lon'] = square['sq_lon']
    features['square_x'] = square['sq_x']
    features['square_y'] = square['sq_y']
    features['time_of_day'] = square['day_hour']
    features['city_code'] = square['city_code']
    
    # quantile
    stat_methods = ["min", "max", "var", "median", "sem", "skew",
                    "sum", "mad", "kurt"]

    #signal strength
    for feature_name in ["SignalStrength", "LocationAltitude", "LocationPrecision",
                         "LocationSpeed", "LocationDirection", "LocationTimestampDelta"]:
        for stat_method in stat_methods:
            features[feature_name + "_" + stat_method] = getattr(group[feature_name], stat_method)()
        
        for i in xrange(1, 10):
            features[feature_name + "_quantile_" + str(i * 10)] = group[feature_name].quantile(i / 10.)
    
    
    """group_by_operator = group.groupby('OperatorID')
    for operator_id in group_by_operator.groups:
        operator_group = group_by_operator.get_group(operator_id)
        for stat_method in stat_methods:
            features["strength_by_operator_" + str(operator_id) + "_" + stat_method] = getattr(operator_group["SignalStrength"], stat_method)()"""
    
    group_by_radio = group.groupby('radio')
    for radio_id in group_by_radio.groups:
        radio_group = group_by_radio.get_group(radio_id)
        for stat_method in stat_methods:
            features["strength_by_radio_" + str(radio_id) + "_" + stat_method] = getattr(radio_group["SignalStrength"], stat_method)()
    
    #features for each user
    group_by_user = group.groupby('u_hashed')
    features['mean_sum_of_var_of_lat_and_lon'] = group_by_user.apply(lambda group: group['ulat'].var()+group['ulon'].var()).mean()
    
    features['num_users'] = len(group_by_user)
    features['mean_entries_per_user'] = group_by_user.apply(len).mean()
    features['mean_user_signal_var'] = group_by_user.apply(
        lambda user_entries: user_entries['SignalStrength'].var()).mean()
    
    #netatmo features
    if square['hour_hash'] in netatmo_groups:
        for k in [1, 2, 3, 5, 10, 20, 40, 50, 70, 89]:
            local_stations,neighbors = netatmo_groups[square['hour_hash']],netatmo_anns[square['hour_hash']]
            [distances],[neighbor_ids] = neighbors.query([(square['sq_lat'],square['sq_lon'])],k=k)

            neighbor_stations = local_stations.iloc[neighbor_ids]

            features['distance_to_closest_station'] = np.min(distances)
            features['mean_distance_to_station_' + str(k)] = np.mean(distances)

            for colname in ['netatmo_pressure_mbar','netatmo_temperature_c','netatmo_sum_rain_24h',
                            'netatmo_humidity_percent',"netatmo_wind_speed_kmh","netatmo_wind_gust_speed_kmh",
                            "sq_x", "sq_y", "netatmo_timestamp_delta", "netatmo_sum_rain_1h", "netatmo_wind_gust_direction_deg",
                           "point_longitude",
                           "netatmo_wind_gust_timestamp", "netatmo_timestamp", "netatmo_wind_direction_deg", "point_latitude",
                           "netatmo_latitude", "netatmo_longitude"]:
                col = neighbor_stations[colname].dropna()
                
                for stat_method in stat_methods:
                    value = np.nan
                    if len(col) != 0:
                        value = getattr(col, stat_method)()
                    features[colname + "_" + stat_method + "_" + str(k)] = value

                for i in xrange(1, 10):
                    value = np.nan
                    if len(col) != 0:
                        value = col.quantile(i / 10.)
                    features[colname + "_quantile_" + str(i * 10) + "_" + str(k)] = value

    return features

groupby = train.groupby(["city_code","sq_x","sq_y","hour_hash"])
groups = groupby.groups


def sleep_30_sec(any_arg):
    import time
    time.sleep(30)
    return any_arg

def extract_all_features():
    def get_group(block_id):
        group = groupby.get_group(block_id)
        return group

    pool = Pool(processes=56, initializer=init_process, initargs=(netatmo_groups, netatmo_anns))
    res = list(pool.imap(sleep_30_sec, xrange(56)))
    group_generator = imap(get_group, groups.keys()[:])
    feature_iterator = pool.imap(extract_features, group_generator)

    X,y,block_ids = [],[],[]
    save_id = 0
    for block_id, features in izip(groups.keys()[:], tqdm(feature_iterator)):
        group = groupby.get_group(block_id)
        X.append(features)
        y.append(group.iloc[0]['rain'])
        block_ids.append(block_id+(group.iloc[0]["hours_since"],))

    X = pd.DataFrame(X)
    y = np.array(y)
    block_ids = pd.DataFrame(block_ids,columns=["city_code","sq_x","sq_y","hour_hash","hours_since"])
    return X,y,block_ids

X,y,block_ids = extract_all_features()

print X.shape

data_to_save = (X,y,block_ids)
with open(CITY_FEATURES_PATH, "wb") as f:
    pickle.dump(data_to_save, f)

