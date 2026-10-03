import os
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

THRESHOLD_EPOCHS = 600
TH_LEARNING_RATE = 0.01
NORMAL_WEIGHT = 1.0
ANOMALY_WEIGHT = 20.0
SMOOTH_WEIGHT = 100.0
THRESHOLD_WEIGHT = 4.0

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
history = mamba_ae.fit(X_train, X_train, validation_data=(X_valid_0, X_valid_0), epochs=100, batch_size=64, shuffle=False, callbacks=[reduce_lr, early_stopping], verbose=1)

# Mamba-AE 가중치 저장
mamba_ae.save_weights('./weights/mamba_ae.weights.h5')
print("Mamba-AE weights saved.")

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
# 12. Latent / Reconstruction Error
# =========================================================
mamba_ae.trainable = False

latent_model = Model(inputs=mamba_ae.input, outputs=mamba_ae.get_layer('latent_vector').output)

valid_latent = latent_model.predict(X_valid, verbose=0)
valid_pred = mamba_ae.predict(X_valid, verbose=0)
valid_error = np.mean(np.square(X_valid - valid_pred), axis=(1, 2))

test_latent = latent_model.predict(X_test, verbose=0)
test_pred = mamba_ae.predict(X_test, verbose=0)
test_error = np.mean(np.square(X_test - test_pred), axis=(1, 2))

# =========================================================
# 13. Adaptive Threshold 설정
# =========================================================

class AdaptiveThreshold(layers.Layer):
    def __init__(self, hidden_dim=16, **kwargs):
        super().__init__(**kwargs)
        self.hidden = layers.Dense(hidden_dim, activation='relu')
        self.threshold = layers.Dense(1, activation='softplus')

    def build(self, input_shape):
        self.temperature_raw = self.add_weight(name='temperature_raw', shape=(), initializer=tf.keras.initializers.Constant(-2.0), trainable=True)
        self.threshold_scale_raw = self.add_weight(name='threshold_scale_raw', shape=(), initializer=tf.keras.initializers.Constant(-1.0), trainable=True)
        super().build(input_shape)

    def call(self, latent):
        x = self.hidden(latent)
        threshold_raw = self.threshold(x)
        threshold_scale = tf.nn.sigmoid(self.threshold_scale_raw)
        threshold = threshold_scale * threshold_raw
        temperature = tf.nn.softplus(self.temperature_raw) + 1e-6
        return threshold, temperature

latent_input = layers.Input(shape=(LATENT_DIM,), name='latent_input')
error_input = layers.Input(shape=(1,), name='error_input')

threshold_layer = AdaptiveThreshold(hidden_dim=16, name='adaptive_threshold_layer')
adaptive_threshold, temperature = threshold_layer(latent_input)

score = layers.Lambda(lambda x: (x[0] - x[1]) / x[2], name='anomaly_score')([error_input, adaptive_threshold, temperature])
anomaly_probability = layers.Activation('sigmoid', name='anomaly_probability')(score)

base_threshold_model = Model(inputs=[latent_input, error_input], outputs=[anomaly_probability, adaptive_threshold], name='Mamba_Window_Threshold_Model')

# =========================================================
# 14. Threshold Custom Model
# =========================================================
class ThresholdModel(tf.keras.Model):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def call(self, inputs, training=False):
        return self.base_model(inputs, training=training)

    def calculate_loss(self, y, y_pred, threshold):
        y = tf.cast(tf.reshape(y, (-1, 1)), tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        loss_normal = -NORMAL_WEIGHT * (1.0 - y) * tf.math.log(1.0 - y_pred)
        loss_anomaly = -ANOMALY_WEIGHT * y * tf.math.log(y_pred)
        classification_loss = tf.reduce_mean(loss_normal + loss_anomaly)
        smooth_loss = tf.reduce_mean(tf.square(threshold[1:] - threshold[:-1]))
        threshold_loss = tf.reduce_mean(threshold)
        total_loss = classification_loss + SMOOTH_WEIGHT * smooth_loss + THRESHOLD_WEIGHT * threshold_loss
        return total_loss, classification_loss, smooth_loss

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred, threshold = self(x, training=True)
            total_loss, classification_loss, smooth_loss = self.calculate_loss(y, y_pred, threshold)
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"loss": total_loss, "classification_loss": classification_loss, "smooth_loss": smooth_loss}

    def test_step(self, data):
        x, y = data
        y_pred, threshold = self(x, training=False)
        total_loss, classification_loss, smooth_loss = self.calculate_loss(y, y_pred, threshold)
        return {"loss": total_loss, "classification_loss": classification_loss, "smooth_loss": smooth_loss}

threshold_model = ThresholdModel(base_threshold_model)
threshold_model.compile(optimizer=Adam(learning_rate=TH_LEARNING_RATE))

# =========================================================
# 15. Threshold Train
# =========================================================
threshold_history = threshold_model.fit([valid_latent, valid_error.reshape(-1, 1)], Y_valid.astype(np.float32), 
                                        epochs=THRESHOLD_EPOCHS, batch_size=len(valid_latent), shuffle=False, 
                                        callbacks=[EarlyStopping(monitor='loss', patience=15, restore_best_weights=True, verbose=1), 
                                                   ReduceLROnPlateau(monitor='loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1)], 
                                        verbose=1)

# =========================================================
# 16. Window Threshold 추출
# =========================================================
threshold_output_model = Model(inputs=latent_input, outputs=adaptive_threshold)
test_threshold = threshold_output_model.predict(test_latent, verbose=0).reshape(-1)

prediction_model = Model(inputs=[latent_input, error_input], outputs=anomaly_probability)
test_probability = prediction_model.predict([test_latent, test_error.reshape(-1, 1)], verbose=0).reshape(-1)

learned_temperature = float(tf.nn.softplus(threshold_layer.temperature_raw).numpy() + 1e-6)
Y_pred = (test_error > test_threshold).astype(int)
Y_test = Y_test.astype(int)

print("Learned Temperature :", learned_temperature)
print("Adaptive Threshold Mean :", np.mean(test_threshold))
print("Adaptive Threshold Std :", np.std(test_threshold))
print("Adaptive Threshold Min :", np.min(test_threshold))
print("Adaptive Threshold Max :", np.max(test_threshold))

# =========================================================
# 17. Evaluation
# =========================================================
print("\n==============================")
print("Mamba-AE + Adaptive Threshold")
print("==============================")
print("Accuracy :", accuracy_score(Y_test, Y_pred))
print("Precision :", precision_score(Y_test, Y_pred, zero_division=0))
print("Recall :", recall_score(Y_test, Y_pred, zero_division=0))
print("F1 Score :", f1_score(Y_test, Y_pred, zero_division=0))
print("ROC-AUC :", roc_auc_score(Y_test, test_probability))
print(classification_report(Y_test, Y_pred, digits=4, zero_division=0))

# =========================================================
# 18. Confusion Matrix
# =========================================================
cm = confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(7, 7))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Mamba-AE Adaptive Threshold Confusion Matrix')
plt.show()

# =========================================================
# 19. Reconstruction Error / Window Threshold
# =========================================================
plt.figure(figsize=(14, 6))
plt.plot(test_error, label='Reconstruction Error')
plt.plot(test_threshold, label='Window Threshold')
plt.xlabel('Sequence Index')
plt.ylabel('Error / Threshold')
plt.title('Mamba-AE Reconstruction Error vs Window Threshold')
plt.legend()
plt.grid()
plt.show()

# =========================================================
# 20. FP / FN
# =========================================================
false_positive_idx = np.where((Y_test == 0) & (Y_pred == 1))[0]
false_negative_idx = np.where((Y_test == 1) & (Y_pred == 0))[0]
print("False Positive :", len(false_positive_idx))
print("False Negative :", len(false_negative_idx))
print("FP Index :", false_positive_idx)
print("FN Index :", false_negative_idx)

# =========================================================
# 21. Threshold Training Loss
# =========================================================
plt.figure(figsize=(10, 5))
plt.plot(threshold_history.history['loss'], label='Threshold Total Loss')
plt.plot(threshold_history.history['classification_loss'], label='Classification Loss')
plt.plot(threshold_history.history['smooth_loss'], label='Smooth Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Mamba Adaptive Threshold Loss')
plt.legend()
plt.grid()
plt.show()