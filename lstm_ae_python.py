import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import Model, models, layers, optimizers, regularizers
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split

normal = pd.read_csv('./press_data_normal.csv', index_col =0)
outlier = pd.read_csv('./outlier_data.csv', index_col = 0)

normal_data = normal.copy()
outlier_data = outlier.copy()

use_col = ['AI0_Vibration', 'AI1_Vibration', 'AI2_Current']

#=================data preprocessing=======================
normal_data[use_col] = normal_data[use_col].abs()
outlier_data[use_col] = outlier_data[use_col].abs()

X_normal = normal_data[use_col]
y_normal = normal_data['Equipment_state']
X_anomaly = outlier_data[use_col]
y_anomaly = outlier_data['Equipment_state']

X_train_normal = X_normal[:15000]
y_train_normal = y_normal[:15000]
X_test_normal = X_normal[15000:]
y_test_normal = y_normal[15000:]
X_test_anomaly = X_anomaly
y_test_anomaly = y_anomaly

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train_normal)
X_test_normal_scaled = scaler.transform(X_test_normal)
X_test_anomaly_scaled = scaler.transform(X_test_anomaly)

y_train_normal = np.array(y_train_normal)
y_test_normal = np.array(y_test_normal)
y_test_anomaly = np.array(y_test_anomaly)
#=============================================================
sequence = 50
#===========================================================
X_train, Y_train = [], []
for index in range(len(X_train_scaled) - sequence - 100):
    X_train.append(X_train_scaled[index: index + sequence])
    Y_train.append(y_train_normal[index + sequence + 100])
    
X_test_normal, Y_test_normal = [], []
for index in range(len(X_test_normal_scaled) - sequence - 100):
    X_test_normal.append(X_test_normal_scaled[index:index+sequence])
    Y_test_normal.append(y_test_normal[index + sequence + 100])
    
X_test_anomal, Y_test_anomal = [], []
for index in range(len(X_test_anomaly_scaled) - sequence - 100):
    X_test_anomal.append(X_test_anomaly_scaled[index:index+sequence])
    Y_test_anomal.append(y_test_anomaly[index + sequence + 100])
    
X_test_normal, Y_test_normal = np.array(X_test_normal), np.array(Y_test_normal)
X_test_anomal, Y_test_anomal = np.array(X_test_anomal), np.array(Y_test_anomal)

X_valid_normal, Y_valid_normal = X_test_normal[:880, :, :], Y_test_normal[:880]
X_test_normal, Y_test_normal = X_test_normal[880:,:,:], Y_test_normal[880:]
X_valid_anomal, Y_valid_anomal = X_test_anomal[:300, :, :], Y_test_anomal[:300]
X_test_anomal, Y_test_anomal = X_test_anomal[300:, :, :], Y_test_anomal[300:]

X_valid = np.vstack((X_valid_normal, X_valid_anomal))
Y_valid = np.hstack((Y_valid_normal, Y_valid_anomal))

X_test = np.vstack((X_test_normal, X_test_anomal))
Y_test = np.hstack((Y_test_normal, Y_test_anomal))

X_train, Y_train = np.array(X_train), np.array(Y_train)
X_valid, Y_valid = np.array(X_valid), np.array(Y_valid)
X_test, Y_test = np.array(X_test), np.array(Y_test)

X_valid_0 = X_valid[Y_valid == 0]

def LSTM_AE(sequence, n_features):
    lstm_ae = models.Sequential()
    
    #encoder
    lstm_ae.add(layers.LSTM(64, input_shape=(sequence, n_features), return_sequences=True))
    lstm_ae.add(layers.LSTM(32, return_sequences=False))
    lstm_ae.add(layers.RepeatVector(sequence))
    
    #decoder
    lstm_ae.add(layers.LSTM(32, return_sequences=True))
    lstm_ae.add(layers.LSTM(64, return_sequences=True))
    lstm_ae.add(layers.TimeDistributed(layers.Dense(n_features)))
    return lstm_ae

lstm_ae = LSTM_AE(sequence, 3)
lstm_ae.summary()

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor = 0.7, patience=50, verbose=1)
es = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=120, verbose=1, mode='min',
                   restore_best_weights=True)

lstm_ae.compile(loss='mse', optimizer=optimizers.Adam(0.001))

history = lstm_ae.fit(X_train, X_train, epochs = 100, batch_size = 128, callbacks = [reduce_lr, es], validation_data = (X_valid_0, X_valid_0))    

plt.plot(history.history['loss'], label = 'train loss')
plt.plot(history.history['val_loss'], label = 'valid loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.show()

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix

# 정상 validation reconstruction error
X_valid_pred = lstm_ae.predict(X_valid_0, verbose=0)
valid_error = np.mean(np.square(X_valid_0 - X_valid_pred), axis=(1, 2))

print("Validation Reconstruction Error")
print("Mean :", np.mean(valid_error))
print("Std  :", np.std(valid_error))
print("Min  :", np.min(valid_error))
print("Max  :", np.max(valid_error))

# threshold
threshold = np.percentile(valid_error, 99)
print("Threshold :", threshold)

# 정상/이상 test reconstruction error
normal_pred = lstm_ae.predict(X_test_normal, verbose=0)
anomaly_pred = lstm_ae.predict(X_test_anomal, verbose=0)

normal_error = np.mean(np.square(X_test_normal - normal_pred), axis=(1, 2))
anomaly_error = np.mean(np.square(X_test_anomal - anomaly_pred), axis=(1, 2))

# Reconstruction Error Distribution
plt.figure(figsize=(12, 5))
plt.hist(normal_error, bins=100, alpha=0.6, label='Normal')
plt.hist(anomaly_error, bins=100, alpha=0.6, label='Anomaly')
plt.axvline(threshold, linestyle='--', label='Threshold')
plt.xlabel('Reconstruction Error')
plt.ylabel('Count')
plt.title('LSTM-AE Reconstruction Error Distribution')
plt.legend()
plt.grid()
plt.show()

# 전체 test reconstruction error
X_test_pred = lstm_ae.predict(X_test, verbose=0)
test_error = np.mean(np.square(X_test - X_test_pred), axis=(1, 2))

# threshold 기반 이상 판정
Y_pred = (test_error > threshold).astype(int)
Y_test_eval = Y_test.astype(int)

# 평가 지표
print("\n==============================")
print("Evaluation")
print("==============================")
print("Accuracy :", accuracy_score(Y_test_eval, Y_pred))
print("Precision :", precision_score(Y_test_eval, Y_pred, zero_division=0))
print("Recall :", recall_score(Y_test_eval, Y_pred, zero_division=0))
print("F1 Score :", f1_score(Y_test_eval, Y_pred, zero_division=0))
print("ROC-AUC :", roc_auc_score(Y_test_eval, test_error))

print("\nClassification Report")
print(classification_report(Y_test_eval, Y_pred, digits=4, zero_division=0))

# confusion matrix
cm = confusion_matrix(Y_test_eval, Y_pred)
print("\nConfusion Matrix")
print(cm)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('LSTM-AE Confusion Matrix')
plt.show()

# 시간 순 anomaly score
plt.figure(figsize=(14, 5))
plt.plot(test_error, label='Reconstruction Error')
plt.axhline(threshold, linestyle='--', label='Threshold')
plt.xlabel('Sequence Index')
plt.ylabel('Reconstruction Error')
plt.title('LSTM-AE Anomaly Score over Time')
plt.legend()
plt.grid()
plt.show()

# FP / FN
false_positive_idx = np.where((Y_test_eval == 0) & (Y_pred == 1))[0]
false_negative_idx = np.where((Y_test_eval == 1) & (Y_pred == 0))[0]

print("False Positive 개수 :", len(false_positive_idx))
print("False Negative 개수 :", len(false_negative_idx))
print("False Positive Index :", false_positive_idx)
print("False Negative Index :", false_negative_idx)

# FP / FN 위치 표시
plt.figure(figsize=(14, 5))
plt.plot(test_error, label='Reconstruction Error')
plt.axhline(threshold, linestyle='--', label='Threshold')

if len(false_positive_idx) > 0:
    plt.scatter(false_positive_idx, test_error[false_positive_idx], label='False Positive')

if len(false_negative_idx) > 0:
    plt.scatter(false_negative_idx, test_error[false_negative_idx], label='False Negative')

plt.xlabel('Sequence Index')
plt.ylabel('Reconstruction Error')
plt.title('LSTM-AE False Positive / False Negative')
plt.legend()
plt.grid()
plt.show()