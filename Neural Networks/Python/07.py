# Pre-processing & Data Encoding
import pandas as pd

# Import dataset
df = pd.read_csv("datasets/anomalous_traffic_non_const_v6.csv", na_values=['NA','?'])

# Convert fields to dummy variables
def dummies_encode(df, fields):
    for i in fields:
        df = pd.concat([df, pd.get_dummies(df[i], prefix = i)], axis = 1)
        df.drop(i, axis = 1, inplace = True)

# Extract the CoAP Payload length into a new column
def coap_payload_length(df):
    df[["coap.payload", "coap.payload.format", "coap.payload_length"]] = df["coap.payload"].str.split(':', expand = True)
    df.drop('coap.payload', axis = 1, inplace = True)
    df.drop('coap.payload.format', axis = 1, inplace = True)

# Fields to encode
fields_to_encode = [
    'wpan.frame_length', 'attack_type', 'icmpv6.rpl.opt.length', 'frame.cap_len',
    'frame.protocols', 'udp.length', 'icmpv6.rpl.dio.version', 'coap.opt.uri_path',
    'coap.opt.length', 'coap.type', 'ipv6.plen', 'frame.len', 'ipv6.nxt', 'icmpv6.rpl.opt.type',
    'coap.payload_length', 'icmpv6.code', 'coap.code', 'icmpv6.rpl.dio.dtsn'
]

coap_payload_length(df)
dummies_encode(df, fields_to_encode)

print(df)

print(f'[DONE] Pre-processing & Data Encoding -- PART 01')
