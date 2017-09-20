
import sys
from multiprocessing.pool import Pool

TEST_PATH = "./data/test_spb_features.tsv"
TEST_NETATMO_PATH = "./data/test_spb_netatmo.tsv"

CITY_FEATURES_PATH = "./saved_test_spb"


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
        

test = pd.read_csv(TEST_PATH, sep='\t',dtype=json.load(open("./data/test_col_dtypes.json")),)
test_groupby = test.groupby(["city_code","sq_x","sq_y","hour_hash"])

netatmo = pd.read_csv(TEST_NETATMO_PATH,na_values="None",sep='\t',dtype={'hour_hash':"uint64"})

test_netatmo_groups,test_netatmo_anns = preprocess_netatmo(netatmo)

test_groups = test_groupby.groups

def extract_all_features_test():
    def get_group(block_id):
        group = test_groupby.get_group(block_id)
        return group

    pool = Pool(processes=56, initializer=init_process, initargs=(test_netatmo_groups,test_netatmo_anns))
    
    group_generator = imap(get_group, test_groups.keys()[:])
    feature_iterator = pool.imap(extract_features, group_generator)

    X,block_ids = [],[]
    
    for block_id, features in izip(test_groups.keys()[:], tqdm(feature_iterator)):
        group = test_groupby.get_group(block_id)
        X.append(features)
        block_ids.append(block_id)

    X = pd.DataFrame(X)
    block_ids = pd.DataFrame(block_ids,columns=["city_code","sq_x","sq_y","hour_hash"])
    return X,block_ids

X_test,test_block_ids = extract_all_features_test()


with open(CITY_FEATURES_PATH, "wb") as f:
    pickle.dump((X_test,test_block_ids), f)


