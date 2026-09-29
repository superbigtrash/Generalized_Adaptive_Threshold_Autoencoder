import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras import Model, layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

np.random.seed(42)
tf.random.set_seed(42)

# =========================================================
# 1. Selective State Space Model
# =========================================================
class SelectiveSSM(layers.Layer):
    def __init__(self, d_inner, d_state=16, **kwargs):
        super().__init__(**kwargs)
        self.d_inner = d_inner
        self.d_state = d_state

        self.delta_proj = layers.Dense(d_inner)
        self.B_proj = layers.Dense(d_state)
        self.C_proj = layers.Dense(d_state)

    def build(self, input_shape):
        self.A_log = self.add_weight(
            name='A_log', shape=(self.d_inner, self.d_state),
            initializer=tf.keras.initializers.RandomNormal(mean=-2.0, stddev=0.5), trainable=True)

        self.D = self.add_weight(name='D', shape=(self.d_inner,), initializer='ones', trainable=True)

        super().build(input_shape)

    def call(self, x):
        # 입력에 따라 delta, B, C 생성
        delta = tf.nn.softplus(self.delta_proj(x))
        B = self.B_proj(x)
        C = self.C_proj(x)

        # 안정적인 상태전이를 위해 A는 음수로 제한
        A = -tf.exp(self.A_log)

        # (batch, time, feature) -> (time, batch, feature)
        x_t = tf.transpose(x, [1, 0, 2])
        delta_t = tf.transpose(delta, [1, 0, 2])
        B_t = tf.transpose(B, [1, 0, 2])
        C_t = tf.transpose(C, [1, 0, 2])

        batch_size = tf.shape(x)[0]

        # hidden state
        initial_state = tf.zeros((batch_size, self.d_inner, self.d_state),dtype=x.dtype)

        initial_output = tf.zeros((batch_size, self.d_inner),dtype=x.dtype)

        def step(previous, elements):
            state, _ = previous
            xt, dt, Bt, Ct = elements

            # discretized A
            dA = tf.exp(dt[:, :, None] * A[None, :, :])

            # discretized B
            dB = dt[:, :, None] * Bt[:, None, :]

            # state update
            new_state = dA * state + dB * xt[:, :, None]

            # output
            yt = tf.reduce_sum(new_state * Ct[:, None, :],axis=-1)

            # skip connection D*x
            yt = yt + self.D[None, :] * xt

            return new_state, yt

        _, outputs = tf.scan(step,(x_t, delta_t, B_t, C_t),initializer=(initial_state, initial_output))

        # (time, batch, feature) -> (batch, time, feature)
        outputs = tf.transpose(outputs,[1, 0, 2])

        return outputs

# =========================================================
# 2. Mamba Block
# =========================================================
class MambaBlock(layers.Layer):
    def __init__(self, d_model=64, d_state=16, expand=2, conv_kernel=3, dropout=0.1, **kwargs):
        super().__init__(**kwargs)

        self.d_model = d_model
        self.d_inner = d_model * expand

        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.in_proj = layers.Dense(self.d_inner * 2)

        self.conv = layers.Conv1D(filters=self.d_inner, kernel_size=conv_kernel, padding='causal', groups=self.d_inner)

        self.ssm = SelectiveSSM(d_inner=self.d_inner, d_state=d_state)

        self.out_proj = layers.Dense(d_model)
        self.dropout = layers.Dropout(dropout)

    def call(self, inputs, training=None):
        residual = inputs

        x = self.norm(inputs)
        xz = self.in_proj(x)
        x, z = tf.split(xz, 2, axis=-1)
        x = self.conv(x)
        x = tf.nn.silu(x)
        x = self.ssm(x)
        z = tf.nn.silu(z)
        x = x * z
        x = self.out_proj(x)
        x = self.dropout(x, training=training)

        return residual + x

# =========================================================
# 3. Mamba Autoencoder
# =========================================================
def build_mamba_autoencoder(sequence_length=50, n_features=3, d_model=64, d_state=16, latent_dim=32, num_mamba_blocks=2, expand=2, dropout=0.1):

    inputs = layers.Input(shape=(sequence_length, n_features), name='input_sequence')

    # Input Projection
    x = layers.Dense(d_model, name='input_projection')(inputs)

    # Mamba Encoder
    for i in range(num_mamba_blocks):
        x = MambaBlock(d_model=d_model, d_state=d_state, expand=expand, dropout=dropout, name=f'mamba_block_{i+1}')(x)

    # Bottleneck
    x = layers.GlobalAveragePooling1D(name='global_average_pooling')(x)
    x = layers.Dense(64, activation='relu', name='encoder_dense')(x)
    latent = layers.Dense(latent_dim, name='latent_vector')(x)

    # Decoder
    x = layers.Dense(sequence_length * d_model, activation='relu', name='decoder_dense')(latent)
    x = layers.Reshape((sequence_length, d_model), name='decoder_reshape')(x)
    x = layers.Dense(64, activation='relu', name='decoder_ffn')(x)
    outputs = layers.Dense(n_features, name='reconstruction')(x)

    return Model(inputs, outputs, name='Mamba_Autoencoder')


# =========================================================
# 4. Dataset Load
# LSTM-AE 가이드북과 동일한 데이터 구성
# =========================================================
normal = pd.read_csv('./press_data_normal.csv', index_col=0)
outlier = pd.read_csv('./outlier_data.csv', index_col=0)

normal_data = normal.copy()
outlier_data = outlier.copy()

use_col = ['AI0_Vibration', 'AI1_Vibration', 'AI2_Current']

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

# =========================================================
# 5. Scaling
# =========================================================
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train_normal)
X_test_normal_scaled = scaler.transform(X_test_normal)
X_test_anomaly_scaled = scaler.transform(X_test_anomaly)

y_train_normal = np.array(y_train_normal)
y_test_normal = np.array(y_test_normal)
y_test_anomaly = np.array(y_test_anomaly)

# =========================================================
# 6. Sequence
# LSTM-AE 가이드북과 동일
# =========================================================
SEQUENCE_LENGTH = 50
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

# =========================================================
# 7. Validation / Test 분리
# =========================================================
X_valid_normal = X_test_normal_seq[:880]
Y_valid_normal = Y_test_normal[:880]

X_test_normal = X_test_normal_seq[880:]
Y_test_normal = Y_test_normal[880:]

X_valid_anomal = X_test_anomal[:300]
Y_valid_anomal = Y_test_anomal[:300]

X_test_anomal = X_test_anomal[300:]
Y_test_anomal = Y_test_anomal[300:]

X_valid = np.vstack((X_valid_normal, X_valid_anomal))
Y_valid = np.hstack((Y_valid_normal, Y_valid_anomal))

X_test = np.vstack((X_test_normal, X_test_anomal))
Y_test = np.hstack((Y_test_normal, Y_test_anomal))

X_valid_0 = X_valid[Y_valid == 0]

print("Train :", X_train.shape)
print("Valid Normal :", X_valid_normal.shape)
print("Valid Anomaly :", X_valid_anomal.shape)
print("Test Normal :", X_test_normal.shape)
print("Test Anomaly :", X_test_anomal.shape)
print("Final Test :", X_test.shape)

# =========================================================
# 8. Mamba-AE 설정
# =========================================================
N_FEATURES = 3
D_MODEL = 64
D_STATE = 16
LATENT_DIM = 32
NUM_MAMBA_BLOCKS = 2
EXPAND = 2
DROPOUT = 0.1

mamba_ae = build_mamba_autoencoder(
    sequence_length=SEQUENCE_LENGTH,
    n_features=N_FEATURES,
    d_model=D_MODEL,
    d_state=D_STATE,
    latent_dim=LATENT_DIM,
    num_mamba_blocks=NUM_MAMBA_BLOCKS,
    expand=EXPAND,
    dropout=DROPOUT
)

mamba_ae.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
mamba_ae.summary()

# =========================================================
# 9. Callback
# =========================================================
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

# =========================================================
# 10. Train
# =========================================================
history = mamba_ae.fit(X_train, X_train, validation_data=(X_valid_0, X_valid_0),
    epochs=100, batch_size=64, shuffle=False, callbacks=[reduce_lr, early_stopping],verbose=1)

# =========================================================
# 11. Training Loss
# =========================================================
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Mamba-AE Training Loss')
plt.legend()
plt.grid()
plt.show()

# =========================================================
# 12. Threshold
# =========================================================
X_valid_pred = mamba_ae.predict(X_valid_0, verbose=0)
valid_error = np.mean(np.square(X_valid_0 - X_valid_pred), axis=(1, 2))

threshold = np.percentile(valid_error, 99)

print("Validation Error Mean :", np.mean(valid_error))
print("Validation Error Std :", np.std(valid_error))
print("Threshold :", threshold)

# =========================================================
# 13. Normal / Anomaly Reconstruction Error
# =========================================================
normal_pred = mamba_ae.predict(X_test_normal, verbose=0)
anomaly_pred = mamba_ae.predict(X_test_anomal, verbose=0)

normal_error = np.mean(np.square(X_test_normal - normal_pred), axis=(1, 2))
anomaly_error = np.mean(np.square(X_test_anomal - anomaly_pred), axis=(1, 2))

plt.figure(figsize=(12, 5))
plt.hist(normal_error, bins=100, alpha=0.6, label='Normal')
plt.hist(anomaly_error, bins=100, alpha=0.6, label='Anomaly')
plt.axvline(threshold, linestyle='--', label='Threshold')
plt.xlabel('Reconstruction Error')
plt.ylabel('Count')
plt.title('Mamba-AE Reconstruction Error Distribution')
plt.legend()
plt.grid()
plt.show()

# =========================================================
# 14. 전체 Test
# =========================================================
X_test_pred = mamba_ae.predict(X_test, verbose=0)
test_error = np.mean(np.square(X_test - X_test_pred), axis=(1, 2))

Y_pred = (test_error > threshold).astype(int)
Y_test = Y_test.astype(int)

# =========================================================
# 15. 평가
# =========================================================
print("\n==============================")
print("Mamba-AE Evaluation")
print("==============================")

print("Accuracy :", accuracy_score(Y_test, Y_pred))
print("Precision :", precision_score(Y_test, Y_pred, zero_division=0))
print("Recall :", recall_score(Y_test, Y_pred, zero_division=0))
print("F1 Score :", f1_score(Y_test, Y_pred, zero_division=0))
print("ROC-AUC :", roc_auc_score(Y_test, test_error))

print("\nClassification Report")
print(classification_report(Y_test, Y_pred, digits=4, zero_division=0))

# =========================================================
# 16. Confusion Matrix
# =========================================================
cm = confusion_matrix(Y_test, Y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Mamba-AE Confusion Matrix')
plt.show()

# =========================================================
# 17. FP / FN
# =========================================================
false_positive_idx = np.where((Y_test == 0) & (Y_pred == 1))[0]
false_negative_idx = np.where((Y_test == 1) & (Y_pred == 0))[0]

print("False Positive :", len(false_positive_idx))
print("False Negative :", len(false_negative_idx))
print("FP Index :", false_positive_idx)
print("FN Index :", false_negative_idx)

# =========================================================
# 18. Anomaly Score
# =========================================================
plt.figure(figsize=(14, 5))
plt.plot(test_error, label='Reconstruction Error')
plt.axhline(threshold, linestyle='--', label='Threshold')

if len(false_positive_idx) > 0:
    plt.scatter(false_positive_idx, test_error[false_positive_idx], label='False Positive')

if len(false_negative_idx) > 0:
    plt.scatter(false_negative_idx, test_error[false_negative_idx], label='False Negative')

plt.xlabel('Sequence Index')
plt.ylabel('Reconstruction Error')
plt.title('Mamba-AE Anomaly Score')
plt.legend()
plt.grid()
plt.show()

# =========================================================
# 19. 결과 표
# =========================================================
result_table = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC', 'False Positive', 'False Negative'],
    'Mamba-AE': [
        accuracy_score(Y_test, Y_pred),
        precision_score(Y_test, Y_pred, zero_division=0),
        recall_score(Y_test, Y_pred, zero_division=0),
        f1_score(Y_test, Y_pred, zero_division=0),
        roc_auc_score(Y_test, test_error), len(false_positive_idx), len(false_negative_idx)]})

print(result_table.to_string(index=False))