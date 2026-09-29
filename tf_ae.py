import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras import Model, layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (confusion_matrix,classification_report,
                            accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score)

# =========================================================
# 0. Seed
# =========================================================
np.random.seed(42)
tf.random.set_seed(42)
# =========================================================
# 1. Positional Encoding
# =========================================================
class PositionalEncoding(layers.Layer):
    def __init__(self, sequence_length, d_model):
        super().__init__()

        position = np.arange(sequence_length)[:, np.newaxis]
        dimension = np.arange(d_model)[np.newaxis, :]

        angle_rates = 1 / np.power(10000, (2 * (dimension // 2)) / np.float32(d_model))
        angle_rads = position * angle_rates
        pos_encoding = np.zeros((sequence_length, d_model))

        # 짝수 index -> sin
        pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # 홀수 index -> cos
        pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])
        self.pos_encoding = tf.cast(pos_encoding[np.newaxis, ...], dtype=tf.float32)

    def call(self, x):
        return (x + self.pos_encoding[:, :tf.shape(x)[1], :])

# =========================================================
# 2. Transformer Encoder Block
# =========================================================
def transformer_encoder(x, d_model, num_heads, ff_dim, dropout=0.1):
    # -----------------------------------------------------
    # Multi-Head Attention
    # -----------------------------------------------------
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(query=x, value=x, key=x)
    attention_output = layers.Dropout(dropout)(attention_output)

    # -----------------------------------------------------
    # Add & Norm
    # -----------------------------------------------------
    x = layers.LayerNormalization(epsilon=1e-6)(x + attention_output)

    # -----------------------------------------------------
    # Feed Forward Network
    # -----------------------------------------------------
    ffn_output = layers.Dense(ff_dim,activation='relu')(x)
    ffn_output = layers.Dense(d_model)(ffn_output)
    ffn_output = layers.Dropout(dropout)(ffn_output)

    # -----------------------------------------------------
    # Add & Norm
    # -----------------------------------------------------
    x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)
    return x

# =========================================================
# 3. Transformer Autoencoder
# =========================================================
def build_transformer_autoencoder(
    sequence_length=50,
    n_features=3,
    d_model=64,
    num_heads=4,
    ff_dim=128,
    latent_dim=16,
    num_encoder_blocks=2,
    dropout=0.1):

    # -----------------------------------------------------
    # Input
    # -----------------------------------------------------
    inputs = layers.Input(shape=(sequence_length, n_features), name='input_sequence')

    # -----------------------------------------------------
    # Input Embedding (sequence, 3)->(sequence, d_model)
    # -----------------------------------------------------
    x = layers.Dense(d_model, name='input_embedding')(inputs)

    # -----------------------------------------------------
    # Positional Encoding
    # -----------------------------------------------------
    x = PositionalEncoding(sequence_length=sequence_length, d_model=d_model)(x)

    # -----------------------------------------------------
    # Transformer Encoder
    # -----------------------------------------------------
    for _ in range(num_encoder_blocks):
        x = transformer_encoder(x=x, d_model=d_model, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout)
        
    # -----------------------------------------------------
    # Bottleneck
    # -----------------------------------------------------
    x = layers.GlobalAveragePooling1D(name='global_average_pooling')(x)
    x = layers.Dense(64, activation='relu', name='encoder_dense')(x)
    latent = layers.Dense(latent_dim, activation=None, name='latent_vector')(x)

    # -----------------------------------------------------
    # Decoder
    # -----------------------------------------------------
    x = layers.Dense(sequence_length * d_model,activation='relu', name='decoder_dense')(latent)
    x = layers.Reshape((sequence_length, d_model), name='decoder_reshape')(x)
    x = layers.Dense(64, activation='relu', name='decoder_ffn')(x)

    # -----------------------------------------------------
    # Reconstruction
    # -----------------------------------------------------
    outputs = layers.Dense(n_features, activation=None, name='reconstruction')(x)
    model = Model(inputs=inputs, outputs=outputs, name='Transformer_Autoencoder')
    return model

# 4. Sequence 생성 함수
def make_sequence(data, labels, sequence_length):
    X, Y = [], []

    for i in range(len(data) - sequence_length + 1):
        X.append(data[i:i + sequence_length])

        # window 마지막 시점의 label
        Y.append(labels[i + sequence_length - 1])
    return (np.array(X), np.array(Y))

# =========================================================
# 5. Dataset Load
# =========================================================
normal = pd.read_csv('./press_data_normal.csv', index_col=0)
outlier = pd.read_csv('./outlier_data.csv', index_col=0)

normal_data = normal.copy()
outlier_data = outlier.copy()

use_col = ['AI0_Vibration', 'AI1_Vibration', 'AI2_Current']

# =========================================================
# 6. Feature / Label
# ========================================================
normal_data[use_col] = normal_data[use_col].abs()
outlier_data[use_col] = outlier_data[use_col].abs()

X_normal = normal_data[use_col]
y_normal = normal_data['Equipment_state']

X_anomaly = outlier_data[use_col]
y_anomaly = outlier_data['Equipment_state']

print("Normal shape :", X_normal.shape)
print("Anomaly shape :", X_anomaly.shape)

# =========================================================
# 7. Train / Test 분리
# =========================================================
X_train_normal = X_normal[:15000]
y_train_normal = y_normal[:15000]

X_test_normal = X_normal[15000:]
y_test_normal = y_normal[15000:]

X_test_anomaly = X_anomaly
y_test_anomaly = y_anomaly

# =========================================================
# 8. Scaling
# 정상 Train 데이터로만 fit
# =========================================================
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train_normal)
X_test_normal_scaled = scaler.transform(X_test_normal)
X_test_anomaly_scaled = scaler.transform(X_test_anomaly)

y_train_normal = np.array(y_train_normal)
y_test_normal = np.array(y_test_normal)
y_test_anomaly = np.array(y_test_anomaly)

# =========================================================
# 9. Sequence 생성
# 가이드북 LSTM-AE와 동일
# =========================================================
SEQUENCE_LENGTH = 20
FUTURE_OFFSET = 100

X_train, Y_train = [], []
for index in range(len(X_train_scaled) - SEQUENCE_LENGTH - FUTURE_OFFSET):
    X_train.append(X_train_scaled[index:index + SEQUENCE_LENGTH])
    Y_train.append(y_train_normal[index + SEQUENCE_LENGTH + FUTURE_OFFSET])

X_test_normal_seq, Y_test_normal = [], []
for index in range(len(X_test_normal_scaled) - SEQUENCE_LENGTH - FUTURE_OFFSET):
    X_test_normal_seq.append(X_test_normal_scaled[index:index + SEQUENCE_LENGTH])
    Y_test_normal.append(y_test_normal[index + SEQUENCE_LENGTH + FUTURE_OFFSET])

X_test_anomal, Y_test_anomal = [], []
for index in range(len(X_test_anomaly_scaled) - SEQUENCE_LENGTH - FUTURE_OFFSET):
    X_test_anomal.append(X_test_anomaly_scaled[index:index + SEQUENCE_LENGTH])
    Y_test_anomal.append(y_test_anomaly[index + SEQUENCE_LENGTH + FUTURE_OFFSET])

X_train = np.array(X_train)
Y_train = np.array(Y_train)

X_test_normal_seq = np.array(X_test_normal_seq)
Y_test_normal = np.array(Y_test_normal)

X_test_anomal = np.array(X_test_anomal)
Y_test_anomal = np.array(Y_test_anomal)

print("X_train :", X_train.shape)
print("X_test_normal :", X_test_normal_seq.shape)
print("X_test_anomal :", X_test_anomal.shape)

# =========================================================
# 10. Validation / Test 분리
# 가이드북과 동일
# =========================================================
X_valid_normal = X_test_normal_seq[:880]
Y_valid_normal = Y_test_normal[:880]

X_test_normal = X_test_normal_seq[880:]
Y_test_normal = Y_test_normal[880:]

X_valid_anomal = X_test_anomal[:300]
Y_valid_anomal = Y_test_anomal[:300]

X_test_anomal = X_test_anomal[300:]
Y_test_anomal = Y_test_anomal[300:]

# Validation 전체
X_valid = np.vstack((X_valid_normal, X_valid_anomal))
Y_valid = np.hstack((Y_valid_normal, Y_valid_anomal))

# Test 전체
X_test = np.vstack((X_test_normal, X_test_anomal))
Y_test = np.hstack((Y_test_normal, Y_test_anomal))

# 정상 Validation만 AE validation loss에 사용
X_valid_0 = X_valid[Y_valid == 0]

print()
print("Train :", X_train.shape)
print("Valid Normal :", X_valid_normal.shape)
print("Valid Anomaly :", X_valid_anomal.shape)
print("Test Normal :", X_test_normal.shape)
print("Test Anomaly :", X_test_anomal.shape)
print("Final Valid :", X_valid.shape)
print("Final Test :", X_test.shape)
print("Valid Normal Only :", X_valid_0.shape)

# =========================================================
# 12. Transformer Autoencoder 설정
# =========================================================
N_FEATURES = 3
D_MODEL = 64
NUM_HEADS = 4
FF_DIM = 128
LATENT_DIM = 32
NUM_ENCODER_BLOCKS = 2
DROPOUT = 0.1

# =========================================================
# 13. Model 생성
# =========================================================
model = build_transformer_autoencoder(
    sequence_length=SEQUENCE_LENGTH,
    n_features=N_FEATURES,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    ff_dim=FF_DIM,
    latent_dim=LATENT_DIM,
    num_encoder_blocks=NUM_ENCODER_BLOCKS, 
    dropout=DROPOUT)
# =========================================================
# 14. Compile
# =========================================================
model.compile(optimizer=Adam(learning_rate=1e-3),loss='mse')
model.summary()

# =========================================================
# 15. Callback
# =========================================================
early_stopping = EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True,verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=5,min_lr=1e-6,verbose=1)

# =========================================================
# 16. Train
# Autoencoder: X_train -> X_train 정상 데이터만 학습
# =========================================================
history = model.fit(X_train,X_train,
                    validation_data=(X_valid_0, X_valid_0),
                    epochs=100, batch_size=64, shuffle=False,
                    callbacks=[early_stopping,reduce_lr], verbose=1)

# =========================================================
# 17. Training Loss Plot
# =========================================================
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Transformer Autoencoder Training Loss')
plt.legend()
plt.grid()
plt.show()
# =========================================================
# 18. 정상 Validation Reconstruction
# =========================================================
X_valid_pred = model.predict(X_valid_normal, verbose=0)
# =========================================================
# 19. 정상 Validation Reconstruction Error
# 각 Window마다 MSE 1개 계산
# =========================================================
valid_error = np.mean(np.square(X_valid_normal - X_valid_pred), axis=(1, 2))

print()
print("Validation reconstruction error")
print("Mean :", np.mean(valid_error))
print("Std  :", np.std(valid_error))
print("Min  :", np.min(valid_error))
print("Max  :", np.max(valid_error))

# =========================================================
# 20. Threshold 설정
# 정상 Validation reconstruction error의
# 상위 1%를 이상으로 간주
# =========================================================
threshold = np.percentile(valid_error, 99)
print()
print("Threshold :", threshold)

# =========================================================
# 21. Normal Test Reconstruction
# =========================================================
normal_pred = model.predict(X_test_normal, verbose=0)
normal_error = np.mean(np.square(X_test_normal - normal_pred), axis=(1, 2))

# =========================================================
# 22. Anomaly Test Reconstruction
# =========================================================
anomaly_pred = model.predict(X_test_anomal, verbose=0)
anomaly_error = np.mean(np.square(X_test_anomal - anomaly_pred), axis=(1, 2))

# =========================================================
# 23. Reconstruction Error 비교
# =========================================================
print()
print("Normal Test Error")
print("Mean :", np.mean(normal_error))
print("Max  :", np.max(normal_error))

print()
print("Anomaly Test Error")
print("Mean :", np.mean(anomaly_error))
print("Min  :", np.min(anomaly_error))

# =========================================================
# 24. Reconstruction Error Distribution
# =========================================================
plt.figure(figsize=(12, 5))
plt.hist(normal_error, bins=100, alpha=0.6, label='Normal')
plt.hist(anomaly_error, bins=100, alpha=0.6, label='Anomaly')
plt.axvline(threshold, linestyle='--', label='Threshold')
plt.xlabel('Reconstruction Error')
plt.ylabel('Count')
plt.title('Reconstruction Error Distribution')
plt.legend()
plt.grid()
plt.show()

# =========================================================
# 25. Test 전체 Reconstruction Error
# =========================================================
X_test_pred = model.predict(X_test, verbose=0)
test_error = np.mean(np.square(X_test - X_test_pred), axis=(1, 2))
# =========================================================
# 26. Threshold 기준 Prediction
# 정상 = 0  이상 = 1
# =========================================================
Y_pred = (test_error > threshold).astype(int)

# =========================================================
# 27. Label 확인
# Equipment_state가  정상=0 / 이상=1 이라는 전제
# =========================================================
Y_test = Y_test.astype(int)

# =========================================================
# 28. 평가
# =========================================================
print()
print("==============================")
print("Evaluation")
print("==============================")

print()
print("Accuracy :", accuracy_score(Y_test, Y_pred))
print("Precision :", precision_score(Y_test,Y_pred, zero_division=0))
print("Recall :", recall_score(Y_test,Y_pred, zero_division=0))
print("F1 Score :", f1_score(Y_test,Y_pred, zero_division=0))

# Reconstruction error 자체를
# anomaly score로 사용
try:
    auc = roc_auc_score(Y_test, test_error)
    print("ROC-AUC :", auc)
    
except ValueError : 
    print("ROC-AUC 계산 불가")

# =========================================================
# 29. Classification Report
# =========================================================
print()
print(classification_report(Y_test, Y_pred, digits=4, zero_division=0))

# =========================================================
# 30. Confusion Matrix
# =========================================================
cm = confusion_matrix(Y_test, Y_pred)
print()
print("Confusion Matrix")
print(cm)

# =========================================================
# 31. Confusion Matrix Plot
# =========================================================
plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha='center', va='center')

plt.xticks([0, 1], ['Normal', 'Anomaly'])
plt.yticks([0, 1], ['Normal', 'Anomaly'])
plt.show()

# =========================================================
# 32. 시간 순 Reconstruction Error Plot
# =========================================================
plt.figure(figsize=(14, 5))
plt.plot(test_error, label='Reconstruction Error')
plt.axhline(threshold, linestyle='--', label='Threshold')
plt.xlabel('Sequence Index')
plt.ylabel('Reconstruction Error')
plt.title('Anomaly Score over Time')
plt.legend()
plt.grid()
plt.show()

# =========================================================
# 추가 평가 시각화
# =========================================================

# False Positive / False Negative
false_positive_idx = np.where((Y_test == 0) & (Y_pred == 1))[0]
false_negative_idx = np.where((Y_test == 1) & (Y_pred == 0))[0]

print("False Positive 개수 :", len(false_positive_idx))
print("False Negative 개수 :", len(false_negative_idx))
print("False Positive Index :", false_positive_idx)
print("False Negative Index :", false_negative_idx)

# Confusion Matrix Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Transformer-AE Confusion Matrix')
plt.show()

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
plt.title('Transformer-AE False Positive / False Negative')
plt.legend()
plt.grid()
plt.show()

# 정상/이상 Reconstruction Error Box Plot
error_df = pd.DataFrame({
    'Reconstruction Error': np.concatenate([normal_error, anomaly_error]),
    'State': ['Normal'] * len(normal_error) + ['Anomaly'] * len(anomaly_error)
})

plt.figure(figsize=(7, 5))
sns.boxplot(data=error_df, x='State', y='Reconstruction Error')
plt.title('Transformer-AE Reconstruction Error Comparison')
plt.grid(axis='y')
plt.show()

# 평가 결과 표
result_table = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC', 'False Positive', 'False Negative'],
    'Transformer-AE': [
        accuracy_score(Y_test, Y_pred),
        precision_score(Y_test, Y_pred, zero_division=0),
        recall_score(Y_test, Y_pred, zero_division=0),
        f1_score(Y_test, Y_pred, zero_division=0),
        roc_auc_score(Y_test, test_error),
        len(false_positive_idx),
        len(false_negative_idx)
    ]
})

print("\nTransformer-AE Evaluation")
print(result_table.to_string(index=False))