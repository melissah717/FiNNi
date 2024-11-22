import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def build_finni(shape):
    """
    build FiNNi's model dynamically using different parameteres
    """
    finni = Sequential([
        Input(shape=(shape,)),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    finni.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return finni

def evolve(X, y, params):
    scaler_X = StandardScaler()
    X = scaler_X.fit_transform(X)
    
    scaler_y = StandardScaler()
    y = scaler_y.fit_transform(y)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    finni = build_finni(X_train.shape[1])

    history = finni.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        verbose=3
    )

    return finni, history, scaler_y, X_val, y_val