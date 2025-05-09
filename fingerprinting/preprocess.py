import numpy as np
import tcbench as tcb
from sklearn.preprocessing import StandardScaler

df = tcb.load_parquet(tcb.DATASETS.MIRAGE19, min_pkts=10)
df = df.drop(columns=['conn_id', 'android_name'])
df = df.rename(columns={'flow_metadata_bf_label': 'label', 
                        'flow_metadata_bf_labeling_type': 'labeling_type',
                        'packet_data_l4_raw_payload': 'raw_payload', 
                        'flow_metadata_bf_l4_payload_bytes': 'payload_bytes',
                        'flow_metadata_bf_duration': 'duration',
                        'timetofirst': 'time_to_first',})
print(df.columns)
print(df.shape)

def process_flow(row):
    feature_vector = []
    pkts_size = np.array(row['pkts_size']) if len(row['pkts_size']) > 0 else np.array([0])
    feature_vector.extend([np.mean(pkts_size), np.std(pkts_size), np.min(pkts_size), np.max(pkts_size)])
    return feature_vector

vectors_baseline = np.array(df.apply(process_flow, axis=1).tolist())
scaler = StandardScaler()
vectors_baseline = scaler.fit_transform(vectors_baseline)

label_to_class = {
    "com.joelapenna.foursquared": 0,
    "com.contextlogic.wish": 1,
    "com.pinterest": 2,
    "com.tripadvisor.tripadvisor": 3,
    "com.groupon": 4,
    "com.accuweather.android": 5,
    "com.waze": 6,
    "com.duolingo": 7,
    "com.viber.voip": 8,
    "com.facebook.katana": 9,
    "de.motain.iliga": 10,
    "it.subito": 11,
    "com.facebook.orca": 12,
    "com.dropbox.android": 13,
    "com.twitter.android": 14,
    "com.google.android.youtube": 15,
    "com.spotify.music": 16,
    "com.iconology.comics": 17,
    "com.trello": 18,
    "air.com.hypah.io.slither": 19,
}

df['class'] = df['label'].map(label_to_class)
df['class'] = df['class'].astype(int)
labels_baseline = df['class'].values

np.save('/home/ev357/tcbench/src/fingerprinting/artifacts-mirage19/baseline_vectors.npy', vectors_baseline)
np.save('/home/ev357/tcbench/src/fingerprinting/artifacts-mirage19/baseline_labels.npy', labels_baseline)