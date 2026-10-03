import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import Model, models, layers, optimizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix

np.random.seed(42)
tf.random.set_seed(42)

def flatten(X):
    flattened = np.empty((X.shape[0], X.shape[2]))
    for i in range(X.shape[0]):
        flattened[i] = X[i, X.shape[1] - 1, :]
    return flattened

# ================= Dataset Load =================
normal = pd.read_csv('./press_data_normal.csv', index_col=0)
outlier = pd.read_csv('./outlier_data.csv', index_col=0)

normal_data = normal.copy()
outlier_data = outlier.copy()

use_col = ['AI0_Vibration', 'AI1_Vibration', 'AI2_Current']

# ================= Data Preprocessing =================
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

# ================= Hyper Parameters =================
SEQUENCE = 20
LATENT_DIM = 32

THRESHOLD_EPOCHS = 2000
TH_LEARNING_RATE = 0.01

NORMAL_WEIGHT = 5.0
ANOMALY_WEIGHT = 15.0
SMOOTH_WEIGHT = 50.0
THRESHOLD_WEIGHT = 0.3

# ================= Sequence =================
X_train, Y_train = [], []

for index in range(len(X_train_scaled) - SEQUENCE - 100):
    X_train.append(X_train_scaled[index:index + SEQUENCE])
    Y_train.append(y_train_normal[index + SEQUENCE + 100])

X_test_normal, Y_test_normal = [], []

for index in range(len(X_test_normal_scaled) - SEQUENCE - 100):
    X_test_normal.append(X_test_normal_scaled[index:index + SEQUENCE])
    Y_test_normal.append(y_test_normal[index + SEQUENCE + 100])

X_test_anomal, Y_test_anomal = [], []

for index in range(len(X_test_anomaly_scaled) - SEQUENCE - 100):
    X_test_anomal.append(X_test_anomaly_scaled[index:index + SEQUENCE])
    Y_test_anomal.append(y_test_anomaly[index + SEQUENCE + 100])

X_train = np.array(X_train)
Y_train = np.array(Y_train)

X_test_normal = np.array(X_test_normal)
Y_test_normal = np.array(Y_test_normal)

X_test_anomal = np.array(X_test_anomal)
Y_test_anomal = np.array(Y_test_anomal)

# ================= Validation / Test Split =================
X_valid_normal = X_test_normal[:880]
Y_valid_normal = Y_test_normal[:880]

X_test_normal = X_test_normal[880:]
Y_test_normal = Y_test_normal[880:]

X_valid_anomal = X_test_anomal[:300]
Y_valid_anomal = Y_test_anomal[:300]

X_test_anomal = X_test_anomal[300:]
Y_test_anomal = Y_test_anomal[300:]

X_valid = np.vstack((X_valid_normal, X_valid_anomal))
Y_valid = np.hstack((Y_valid_normal, Y_valid_anomal))

X_test = np.vstack((X_test_normal, X_test_anomal))
Y_test = np.hstack((Y_test_normal, Y_test_anomal))

print("X_train :", X_train.shape)
print("X_valid :", X_valid.shape)
print("X_test :", X_test.shape)
print("Valid Normal :", np.sum(Y_valid == 0))
print("Valid Anomaly :", np.sum(Y_valid == 1))
print("Test Normal :", np.sum(Y_test == 0))
print("Test Anomaly :", np.sum(Y_test == 1))

# ================= LSTM Autoencoder =================
def LSTM_AE(sequence, n_features):
    lstm_ae = models.Sequential()

    # Encoder
    lstm_ae.add(layers.LSTM(64, input_shape=(sequence, n_features), return_sequences=True))
    lstm_ae.add(layers.LSTM(32, return_sequences=False))
    lstm_ae.add(layers.RepeatVector(sequence))

    # Decoder
    lstm_ae.add(layers.LSTM(32, return_sequences=True))
    lstm_ae.add(layers.LSTM(64, return_sequences=True))
    lstm_ae.add(layers.TimeDistributed(layers.Dense(n_features)))

    return lstm_ae

# ================= LSTM-AE Weight Load =================
# AE는 여기서 학습하지 않음
lstm_ae = LSTM_AE(SEQUENCE, 3)
lstm_ae.build((None, SEQUENCE, 3))
lstm_ae.load_weights('./weights/seq_50_lstm_ae.weights.h5')

print("LSTM-AE weight loaded")
lstm_ae.summary()

# ================= Latent / Reconstruction Error =================
# Encoder의 두 번째 LSTM(32) 출력을 latent vector로 사용
latent_input_tensor = layers.Input(shape=(SEQUENCE, 3))
x = lstm_ae.layers[0](latent_input_tensor)
latent_output = lstm_ae.layers[1](x)

latent_model = Model(inputs=latent_input_tensor, outputs=latent_output)

valid_latent = latent_model.predict(X_valid, verbose=0)
valid_pred = lstm_ae.predict(X_valid, verbose=0)
valid_error = np.mean(np.power(flatten(X_valid) - flatten(valid_pred), 2), axis=1)

test_latent = latent_model.predict(X_test, verbose=0)
test_pred = lstm_ae.predict(X_test, verbose=0)
test_error = np.mean(np.power(flatten(X_test) - flatten(test_pred), 2), axis=1)

print("Valid Latent :", valid_latent.shape)
print("Test Latent :", test_latent.shape)

# ================= Adaptive Threshold =================
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

# ================= Threshold Base Model =================
latent_input = layers.Input(shape=(LATENT_DIM,), name='latent_input')
error_input = layers.Input(shape=(1,), name='error_input')
threshold_layer = AdaptiveThreshold(hidden_dim=16, name='adaptive_threshold_layer')
adaptive_threshold, temperature = threshold_layer(latent_input)
score = layers.Lambda(lambda x: (x[0] - x[1]) / x[2], name='anomaly_score')([error_input, adaptive_threshold, temperature])

anomaly_probability = layers.Activation('sigmoid', name='anomaly_probability')(score)
base_threshold_model = Model(inputs=[latent_input, error_input], outputs=[anomaly_probability, adaptive_threshold], name='Window_Threshold_Model')
base_threshold_model.summary()

# ================= Custom Threshold Model =================
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
        total_loss = (classification_loss + SMOOTH_WEIGHT * smooth_loss + THRESHOLD_WEIGHT * threshold_loss)

        return total_loss, classification_loss, smooth_loss, threshold_loss

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred, threshold = self(x, training=True)
            total_loss, classification_loss, smooth_loss, threshold_loss = self.calculate_loss(y, y_pred, threshold)

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {
            "loss": total_loss,
            "classification_loss": classification_loss,
            "smooth_loss": smooth_loss,
            "threshold_loss": threshold_loss
        }

# ================= Threshold Model Train =================
threshold_model = ThresholdModel(base_threshold_model)
threshold_model.compile(optimizer=optimizers.Adam(learning_rate=TH_LEARNING_RATE))
threshold_history = threshold_model.fit([valid_latent, valid_error.reshape(-1, 1)],
    Y_valid.astype(np.float32),epochs=THRESHOLD_EPOCHS, batch_size=len(valid_latent), shuffle=False,
    callbacks=[
        EarlyStopping(monitor='loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1)], verbose=1)

threshold_model.save_weights('./weights/lstm_threshold.weights.h5')
print("Threshold weight saved")

# ================= Test Threshold =================
threshold_output_model = Model(inputs=latent_input, outputs=[adaptive_threshold, temperature])
test_threshold, learned_temperature = threshold_output_model.predict(test_latent, verbose=0)
test_threshold = test_threshold.reshape(-1)

# Reconstruction Error > Local Threshold -> Anomaly
Y_pred = (test_error > test_threshold).astype(int)
prediction_model = Model(inputs=[latent_input, error_input], outputs=anomaly_probability)
test_probability = prediction_model.predict([test_latent, test_error.reshape(-1, 1)], verbose=0).reshape(-1)

# ================= Threshold Information =================
print("Learned Temperature :", learned_temperature)
print("Adaptive Threshold Mean :", np.mean(test_threshold))
print("Adaptive Threshold Std :", np.std(test_threshold))
print("Adaptive Threshold Min :", np.min(test_threshold))
print("Adaptive Threshold Max :", np.max(test_threshold))

# ================= Evaluation =================
print("\n==============================")
print("LSTM-AE Adaptive Threshold Evaluation")
print("==============================")

print("Accuracy :", accuracy_score(Y_test, Y_pred))
print("Precision :", precision_score(Y_test, Y_pred, zero_division=0))
print("Recall :", recall_score(Y_test, Y_pred, zero_division=0))
print("F1 Score :", f1_score(Y_test, Y_pred, zero_division=0))
print("ROC-AUC :", roc_auc_score(Y_test, test_probability))
print(classification_report(Y_test, Y_pred, digits=4, zero_division=0))

# ================= Confusion Matrix =================
cm = confusion_matrix(Y_test, Y_pred)

plt.figure(figsize=(7, 7))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('LSTM-AE Adaptive Threshold Confusion Matrix')
plt.show()

# ================= Reconstruction Error / Threshold =================
plt.figure(figsize=(14, 6))
plt.plot(test_error, label='Reconstruction Error')
plt.plot(test_threshold, label='Window Threshold')
plt.xlabel('Sequence Index')
plt.ylabel('Error / Threshold')
plt.title('LSTM-AE Reconstruction Error vs Adaptive Threshold')
plt.legend()
plt.grid()
plt.show()

# ================= Threshold Loss =================
plt.figure(figsize=(10, 5))
plt.plot(threshold_history.history['loss'], label='Total Loss')
plt.plot(threshold_history.history['classification_loss'], label='Classification Loss')
plt.plot(threshold_history.history['smooth_loss'], label='Smooth Loss')
plt.plot(threshold_history.history['threshold_loss'], label='Threshold Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Adaptive Threshold Loss')
plt.legend()
plt.grid()
plt.show()

# ================= FP / FN =================
false_positive_idx = np.where((Y_test == 0) & (Y_pred == 1))[0]
false_negative_idx = np.where((Y_test == 1) & (Y_pred == 0))[0]

print("False Positive :", len(false_positive_idx))
print("False Negative :", len(false_negative_idx))