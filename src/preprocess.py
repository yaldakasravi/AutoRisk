import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def load_data(path):
    return pd.read_csv(path)

def encode_features(df, categorical_features):
    encoder = OneHotEncoder(sparse=False, drop='first')
    encoded = encoder.fit_transform(df[categorical_features])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())
    df = df.drop(columns=categorical_features)
    df = pd.concat([df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
    return df

def preprocess_data(path):
    df = load_data(path)
    df['vehicle_type'] = df['vehicle_type'].astype('category')
    df['location_risk_score'] = df['location_risk_score'].astype(float)
    categorical_features = ['vehicle_type']
    df = encode_features(df, categorical_features)
    return df
