from sklearn.preprocessing import OneHotEncoder


def one_hot_func(encoded):
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = encoded.reshape(len(encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded