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
labels_baseline = df['label'].factorize()[0]

np.save('/home/ev357/tcbench/src/fingerprinting/artifacts-mirage19/baseline_vectors.npy', vectors_baseline)
np.save('/home/ev357/tcbench/src/fingerprinting/artifacts-mirage19/baseline_labels.npy', labels_baseline)