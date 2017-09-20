import numpy as np
import pandas as pd
import pickle

with open("saved_data_spb", "rb") as f:
    train_spb = pickle.load(f)

with open("saved_data_msk", "rb") as f:
    train_msk = pickle.load(f)

with open("saved_data_kazan", "rb") as f:
    train_kazan = pickle.load(f)

with open("saved_data_spb_radius2", "rb") as f:
    train_spb_radius = pickle.load(f)

with open("saved_data_msk_radius2", "rb") as f:
    train_msk_radius = pickle.load(f)

with open("saved_data_kazan_radius2", "rb") as f:
    train_kazan_radius = pickle.load(f)


with open("save_test_spb", "rb") as f:
    test_spb = pickle.load(f)

with open("save_test_msk", "rb") as f:
    test_msk = pickle.load(f)

with open("save_test_kazan", "rb") as f:
    test_kazan = pickle.load(f)


with open("save_test_spb_radius2", "rb") as f:
    test_spb_radius = pickle.load(f)

with open("save_test_msk_radius2", "rb") as f:
    test_msk_radius = pickle.load(f)

with open("save_test_kazan_radius2", "rb") as f:
    test_kazan_radius = pickle.load(f)


X = pd.concat([train_spb[0], train_msk[0], train_kazan[0]], ignore_index=True)
X_with_radius = pd.concat([train_spb_radius, train_msk_radius, train_kazan_radius], ignore_index=True)


X = pd.concat([X, X_with_radius], axis=1)


y = np.concatenate([train_spb[1], train_msk[1], train_kazan[1]])

block_ids = pd.concat([train_spb[2], train_msk[2], train_kazan[2]], ignore_index=True)


X_test = pd.concat([test_spb[0], test_msk[0], test_kazan[0]], ignore_index=True)
X_test_with_radius = pd.concat([test_spb_radius, test_msk_radius, test_kazan_radius], ignore_index=True)

X_test = pd.concat([X_test, X_test_with_radius], axis=1)

test_columns = X_test.columns
X = X[test_columns]


in_train = block_ids['hours_since'] <= np.percentile(block_ids['hours_since'],85) #leave last 15% for validation

X_train,y_train = X[in_train],y[in_train]
X_val,y_val = X[~in_train],y[~in_train]
print("Training samples: %i; Validation samples: %i"%(len(X_train),len(X_val)))

from catboost import CatBoostClassifier
model = CatBoostClassifier(iterations=500, thread_count=32, random_seed=42,
                           eval_metric="AUC", nan_mode="Min", one_hot_max_size=10).fit(X_train,y_train, [list(X.columns).index("city_code")], verbose=True, eval_set=(X_val, y_val))


model = CatBoostClassifier(iterations=500, thread_count=32, random_seed=42, eval_metric="AUC", nan_mode="Min", one_hot_max_size=10).fit(X,y, [list(X.columns).index("city_code")], verbose=True)

test_block_ids = pd.concat([test_spb[1], test_msk[1], test_kazan[1]], ignore_index=True)

predictions = test_block_ids.copy()
predictions["prediction"] = model.predict_proba(X_test)[:,1]

blocks = pd.read_csv("./data/hackathon_tosubmit.tsv",sep='\t')
assert len(predictions) == len(blocks),"Predictions don't match blocks. Sumbit at your own risk."

merged = pd.merge(blocks,predictions,how='left',on=["sq_x","sq_y","hour_hash"])
assert not np.isnan(merged.prediction).any(), "some predictions are missing. Sumbit at your own risk."


merged[['id','prediction']].to_csv("submit.csv",sep=',',index=False,header=False)

