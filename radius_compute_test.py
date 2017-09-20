import sys
from multiprocessing.pool import Pool
from tqdm import tqdm
from six.moves import zip as izip, map as imap

TEST_NETATMO_PATH = "./data/test_spb_netatmo.tsv"

CITY_FEATURES_PATH = "./save_test_spb"
CITY_RADIUS_FEATURES_PATH = "./save_test_spb_radius"

import numpy as np
import pandas as pd
import json
import pickle

from tqdm import tqdm
from collections import defaultdict

netatmo = pd.read_csv(TEST_NETATMO_PATH,na_values="None", sep='\t',dtype={'hour_hash':"uint64"})
basic_net_data = netatmo[['netatmo_uid', 'netatmo_longitude', 'netatmo_latitude', 'hour_hash']].drop_duplicates()

import math
def get_angle(lat1, long1, lat2, long2):
    dLon = (long2 - long1)
    y = math.sin(dLon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dLon)
    brng = math.atan2(y, x)
    brng = math.degrees(brng)
    brng = (brng + 360) % 360
    brng = 360 - brng
    return brng


from math import radians, cos, sin, asin, sqrt
def get_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = imap(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6367 * c
    return km


def get_normal_id(my_lat, my_lon, radius, basic_net_data, hour_id):
    distances = []

    for uid, lon, lat, hour in basic_net_data[basic_net_data['hour_hash'] == hour_id].values:
        distances.append([get_angle(my_lat, my_lon, lat, lon), get_distance(my_lat, my_lon, lat, lon), uid])
    
    distances = sorted(distances)
    bins = [[] for i in range(6)]
    bins_flag = [False] * 6
    

    for elem in distances:
        angle, dist, uid = elem
        bins[int(angle) // 60].append([dist, uid])
        bins_flag[int(angle) // 60] = True

    
    ans = []
    
    for cur_bin, i in izip(bins, range(6)):
        cur_bin = sorted(cur_bin)
        if bins_flag[i]:
            ans.append(cur_bin[0][1])
            if len(cur_bin) > 1:
                ans.append(cur_bin[1][1])
            if len(cur_bin) > 2:
                ans.append(cur_bin[2][1])
            
    ost = 18 - len(ans)
    
    if ost < len(distances):
        for i in range(ost):
            ans.append(distances[i][2])
    
    return ans

with open(CITY_FEATURES_PATH, "rb") as f:
    test = pickle.load(f)

X_test = test[0]
test_block_ids = test[1]

stat_methods = ["min", "max", "mean", "var", "median", "sem", "skew", "sum", "mad", "kurt"]


def init_netatmo(basic_net_data, block_ids, netatmo):
    get_netatmo_features.args = basic_net_data, block_ids, netatmo

def get_netatmo_features(args):
    gps, hour_id = args
    basic_net_data, block_ids, netatmo = get_netatmo_features.args
    round_features = {}
    my_lat, my_lon = gps
    rad_near_id = get_normal_id(my_lat, my_lon, 150, basic_net_data, hour_id)
    
    for station_id, ind in izip(rad_near_id, range(len(rad_near_id))):
        cur_data = netatmo[(netatmo['hour_hash'] == hour_id) & (netatmo['netatmo_uid'] == station_id)]
        
        net_gps = cur_data[['netatmo_latitude', 'netatmo_longitude']].values
        if len(net_gps) != 0:
            net_lat, net_lon = cur_data[['netatmo_latitude', 'netatmo_longitude']].values[0]
            round_features['net_dist_' + str(ind)] = get_distance(my_lat, my_lon, net_lat, net_lon)
        else:
            round_features['net_dist_' + str(ind)] = np.nan

        
        for colname in ['netatmo_pressure_mbar','netatmo_temperature_c','netatmo_sum_rain_24h', "sq_x", "sq_y", "point_longitude", "point_latitude", "netatmo_latitude", "netatmo_longitude",
                        'netatmo_humidity_percent',"netatmo_wind_speed_kmh","netatmo_wind_gust_speed_kmh"]:
            
            col = cur_data[colname].dropna()
        
            for stat_method in stat_methods:
                value = np.nan
                if len(col) != 0:
                    value = getattr(col, stat_method)()
                round_features[colname + "_" + stat_method + "_" + str(ind) + "_radius"] = value

            for i in range(1, 10):
                value = np.nan
                if len(col) != 0:
                    value = col.quantile(i / 10.)
                round_features[colname + "_quantile_" + str(i * 10) + "_" + str(ind) + "_radius"] = value

    return round_features

def gen_gps_hour(X, block_ids):
    for gps, hour_id in izip(X[['square_lat', 'square_lon']].values, block_ids['hour_hash']):
        yield gps, hour_id

pool = Pool(56, initializer=init_netatmo, initargs=(basic_net_data, test_block_ids, netatmo))

feat_gen = pool.imap(get_netatmo_features, gen_gps_hour(X_test, test_block_ids))
all_feat_test = list(tqdm(feat_gen))
all_feat_test_pd = pd.DataFrame(all_feat_test)


with open(CITY_RADIUS_FEATURES_PATH, "wb") as f:
    pickle.dump(all_feat_pd, f)

