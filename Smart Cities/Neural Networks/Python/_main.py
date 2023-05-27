# --- Options --- #
# Tensorflow logging: OFF
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Main Code --- #
# Pre-processing & Data Encoding
import numpy as np
import pandas as pd
import tensorflow.keras
from sklearn import metrics
from scipy.stats import zscore
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Activation

# Pre-processing & Data Encoding

# Import dataset
df = pd.read_csv("Datasets/Anomalous_Traffic_VF.csv")

# Fields for dummy encode
dummy_fields = [
    'frame.protocols', 'coap.payload_length', 'coap.opt.uri_path', 'coap.opt.length', 'ipv6.nxt',
    'icmpv6.rpl.opt.type', 'coap.code'
]

# Fields to fill with 0
fill_fields_0 = [
    'icmpv6.type', 'icmpv6.rpl.opt.length', 'icmpv6.checksum.status', 'udp.length', 'icmpv6.rpl.dio.version',
    'icmpv6.rpl.dio.rank', 'ipv6.plen', 'frame.len', 'icmpv6.rpl.dio.dtsn'
]

# Fields to fill with -1
fill_fields_1 = [
    '6lowpan.iphc.m', 'coap.type', '6lowpan.iphc.nh', 'icmpv6.code', 'icmpv6.rpl.opt.transit.pathlifetime'
]

# Fields to zscore normalization
zscore_fields = [
    'wpan.frame_length', 'frame.cap_len', 'icmpv6.rpl.opt.length', 'udp.length', 'icmpv6.rpl.dio.version',
    'icmpv6.rpl.dio.rank', 'ipv6.plen', 'frame.len', 'icmpv6.rpl.dio.dtsn'
]

# Convert fields to dummy variables
def dummy_encode(df, fields):
    for i in fields:
        df = pd.concat([df, pd.get_dummies(df[i], prefix = i)], axis = 1)
        df.drop(i, axis = 1, inplace = True)
    return df

# Fill empty cells with n value
def fill_fields(df, fields, n):
    for i in fields:
        df[i] = df[i].fillna(n)
    return df

# Extract the CoAP Payload length into a new column
def coap_payload_length(df):
    df[["coap.payload", "coap.payload.format", "coap.payload_length"]] = df["coap.payload"].str.split(':', expand = True)
    df["coap.payload_length"] = df["coap.payload_length"].fillna(0)
    df.drop('coap.payload', axis = 1, inplace = True)
    df.drop('coap.payload.format', axis = 1, inplace = True)
    return df

# Zscore normalization
def zscore_normalization(df, fields):
    for i in fields:
        df[i] = zscore(df[i])
    return df

fill_fields(df, fill_fields_0, 0)
fill_fields(df, fill_fields_1, -1)

df['icmpv6.type'] = df['icmpv6.type'].replace(155, 1)
df['coap.type'] = df['coap.type'].replace(2, 1)
df['icmpv6.rpl.opt.transit.pathlifetime'] = df['icmpv6.rpl.opt.transit.pathlifetime'].replace(30, 1)

coap_payload_length(df)
zscore_normalization(df, zscore_fields)
df = dummy_encode(df, dummy_fields)

print(f'[DONE] Pre-processing & Data Encoding -- PART 01')

# Classification for different ports range

# 1 - Well-Known Ports
# 2 - Registered Ports
# 3 - Private or Dynamic Ports

def src_port_range(port):
    if port['prt_src'] < 1024:
        return 1
    if port['prt_src'] < 49151:
        return 2
    if port['prt_src'] < 65535:
        return 3

def dst_port_range(port):
    if port['prt_dst'] < 1024:
        return 1
    if port['prt_dst'] < 49151:
        return 2
    if port['prt_dst'] < 65535:
        return 3

df['prt_src'] = df['udp.srcport']
df['prt_dst'] = df['udp.dstport']

df['src_port'] = df.apply (lambda row: src_port_range(row), axis = 1)
df = pd.concat([df, pd.get_dummies(df['src_port'], prefix="src_port_range")], axis = 1)
df.drop('prt_src', axis = 1, inplace = True)
df.drop('udp.srcport', axis = 1, inplace = True)
df.drop('src_port', axis = 1, inplace = True)

df['dst_port'] = df.apply (lambda row: dst_port_range(row), axis = 1)
df = pd.concat([df, pd.get_dummies(df['dst_port'], prefix = "dst_port_range")], axis = 1)
df.drop('prt_dst', axis = 1, inplace = True)
df.drop('udp.dstport', axis = 1, inplace = True)
df.drop('dst_port', axis = 1, inplace = True)

print(f'[DONE] Pre-processing & Data Encoding -- PART 02')

# Convert to Numpy Multiclass Classification
x_columns = df.columns.drop('is_malicious')
x = df[x_columns].values
dummies = pd.get_dummies(df['is_malicious'])
attack = dummies.columns
y = dummies.values

print(f'[DONE] Numpy Multiclass Classification')

# Training validation splitting
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 42)

print(f'[DONE] Training validation splitting')

# Neural Network Model
model = Sequential()
model.add(Dense(50, input_dim = x.shape[1], activation = 'relu')) # Hidden 1
model.add(Dense(25, activation = 'relu')) # Hidden 2
model.add(Dense(y.shape[1], activation = 'softmax')) # Output
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

print(f'[DONE] Neural Network Model')

# Early Stopping
monitor = EarlyStopping(monitor = 'val_loss', min_delta = 1e-3, patience = 10, verbose = 1, mode = 'auto', restore_best_weights = True)
model.fit(x_train, y_train, validation_data = (x_test, y_test), callbacks = [monitor], verbose = 2, epochs = 1000)

print(f'[DONE] Early Stopping')

# Prediction
pred = model.predict(x_test)

# Metrics for the classification
def compute_metrics(pred, y_test):
    predict_classes = np.argmax(pred, axis = 1)
    expected_classes = np.argmax(y_test, axis = 1)

    correct = metrics.accuracy_score(expected_classes, predict_classes)
    print(f"Accuracy: {correct}")

    recall = metrics.recall_score(expected_classes, predict_classes, average = 'weighted')
    print(f"Recall: {recall}")

    precision = metrics.precision_score(expected_classes, predict_classes, average = 'weighted')
    print(f"Precision: {precision}")

    f1score = metrics.f1_score(expected_classes, predict_classes, average = 'weighted')
    print(f"F1Score: {f1score}")

compute_metrics(pred, y_test)
