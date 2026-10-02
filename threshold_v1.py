import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras import Model, layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

np.random.seed(42)
tf.random.set_seed(42)

# =========================================================
# Transformer AE 구조
# =========================================================

class PositionalEncoding(layers.Layer):
    def __init__(self, sequence_length, d_model):
        super().__init__()
        position = np.arange(sequence_length)[:, np.newaxis]
        dimension = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (dimension // 2)) / np.float32(d_model))
        angle_rads = position * angle_rates

        pos_encoding = np.zeros((sequence_length, d_model))
        pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
        pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])

        self.pos_encoding = tf.cast(pos_encoding[np.newaxis, ...], dtype=tf.float32)

    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]


def transformer_encoder(x, d_model, num_heads, ff_dim, dropout=0.1):
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(query=x, value=x, key=x)
    attention_output = layers.Dropout(dropout)(attention_output)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attention_output)

    ffn_output = layers.Dense(ff_dim, activation='relu')(x)
    ffn_output = layers.Dense(d_model)(ffn_output)
    ffn_output = layers.Dropout(dropout)(ffn_output)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)

    return x


def build_transformer_autoencoder(sequence_length=50, n_features=3, d_model=64, num_heads=4, ff_dim=128, latent_dim=32, num_encoder_blocks=2, dropout=0.1):
    inputs = layers.Input(shape=(sequence_length, n_features), name='input_sequence')

    x = layers.Dense(d_model, name='input_embedding')(inputs)
    x = PositionalEncoding(sequence_length, d_model)(x)

    for _ in range(num_encoder_blocks):
        x = transformer_encoder(x, d_model, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(name='global_average_pooling')(x)
    x = layers.Dense(64, activation='relu', name='encoder_dense')(x)
    latent = layers.Dense(latent_dim, activation=None, name='latent_vector')(x)

    x = layers.Dense(sequence_length * d_model, activation='relu', name='decoder_dense')(latent)
    x = layers.Reshape((sequence_length, d_model), name='decoder_reshape')(x)
    x = layers.Dense(64, activation='relu', name='decoder_ffn')(x)

    outputs = layers.Dense(n_features, activation=None, name='reconstruction')(x)

    return Model(inputs, outputs, name='Transformer_Autoencoder')


# =========================================================
# 설정
# =========================================================

SEQUENCE_LENGTH = 50
FUTURE_OFFSET = 100
TH_LEARNING_RATE = 0.01

N_FEATURES = 3
D_MODEL = 64
NUM_HEADS = 4
FF_DIM = 128
LATENT_DIM = 32
NUM_ENCODER_BLOCKS = 2
DROPOUT = 0.1

THRESHOLD_EPOCHS = 800

NORMAL_WEIGHT = 1.0
ANOMALY_WEIGHT = 20.0

SMOOTH_WEIGHT = 100
THRESHOLD_WEIGHT = 4.0
# =========================================================
# 데이터
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

# Scaling
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train_normal)
X_test_normal_scaled = scaler.transform(X_test_normal)
X_test_anomaly_scaled = scaler.transform(X_test_anomaly)

y_train_normal = np.array(y_train_normal)
y_test_normal = np.array(y_test_normal)
y_test_anomaly = np.array(y_test_anomaly)

# Sequence
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


# Validation / Test - Block Split
GAP = SEQUENCE_LENGTH + FUTURE_OFFSET

NORMAL_TRAIN_SIZE = 880
ANOMALY_TRAIN_SIZE = 100

X_valid_normal = X_test_normal_seq[:NORMAL_TRAIN_SIZE]
Y_valid_normal = Y_test_normal[:NORMAL_TRAIN_SIZE]

X_test_normal = X_test_normal_seq[NORMAL_TRAIN_SIZE + GAP:]
Y_test_normal = Y_test_normal[NORMAL_TRAIN_SIZE + GAP:]

X_valid_anomal = X_test_anomal[:ANOMALY_TRAIN_SIZE]
Y_valid_anomal = Y_test_anomal[:ANOMALY_TRAIN_SIZE]

X_test_anomal = X_test_anomal[ANOMALY_TRAIN_SIZE + GAP:]
Y_test_anomal = Y_test_anomal[ANOMALY_TRAIN_SIZE + GAP:]

X_valid = np.vstack((X_valid_normal, X_valid_anomal))
Y_valid = np.hstack((Y_valid_normal, Y_valid_anomal))

normal_part1 = X_test_normal[:1000]
normal_part2 = X_test_normal[1000:2000]
normal_part3 = X_test_normal[2000:3000]
normal_part4 = X_test_normal[3000:]

normal_y1 = Y_test_normal[:1000]
normal_y2 = Y_test_normal[1000:2000]
normal_y3 = Y_test_normal[2000:3000]
normal_y4 = Y_test_normal[3000:]

anomaly_part1 = X_test_anomal[:50]
anomaly_part2 = X_test_anomal[50:100]
anomaly_part3 = X_test_anomal[100:150]

anomaly_y1 = Y_test_anomal[:50]
anomaly_y2 = Y_test_anomal[50:100]
anomaly_y3 = Y_test_anomal[100:150]

X_test = np.vstack((normal_part1, anomaly_part1, normal_part2, anomaly_part2, normal_part3, anomaly_part3, normal_part4))
Y_test = np.hstack((normal_y1, anomaly_y1, normal_y2, anomaly_y2, normal_y3, anomaly_y3, normal_y4))

print("Scenario Test :", X_test.shape)
print("Normal :", np.sum(Y_test == 0))
print("Anomaly :", np.sum(Y_test == 1))

# AE 생성 + 기존 Weight Load  여기서는 AE 학습 안 함
model = build_transformer_autoencoder(
    sequence_length=SEQUENCE_LENGTH,
    n_features=N_FEATURES,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    ff_dim=FF_DIM,
    latent_dim=LATENT_DIM,
    num_encoder_blocks=NUM_ENCODER_BLOCKS,
    dropout=DROPOUT)

model.load_weights('./weights/tf_ae.weights.h5')

print("AE weight loaded")

# Latent / Reconstruction Error
latent_model = Model(inputs=model.input, outputs=model.get_layer('latent_vector').output)

valid_latent = latent_model.predict(X_valid, verbose=0)
valid_pred = model.predict(X_valid, verbose=0)
valid_error = np.mean(np.square(X_valid - valid_pred), axis=(1, 2))

test_latent = latent_model.predict(X_test, verbose=0)
test_pred = model.predict(X_test, verbose=0)
test_error = np.mean(np.square(X_test - test_pred), axis=(1, 2))

# Adaptive Threshold
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

base_threshold_model = Model(inputs=[latent_input, error_input], outputs=[anomaly_probability, adaptive_threshold], name='Window_Threshold_Model')

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
        classification_loss = tf.reduce_mean(loss_normal + loss_anomaly)
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

threshold_history = threshold_model.fit([valid_latent, valid_error.reshape(-1, 1)], Y_valid.astype(np.float32), epochs=THRESHOLD_EPOCHS, batch_size=len(valid_latent), shuffle=False, callbacks=[EarlyStopping(monitor='loss', patience=15, restore_best_weights=True, verbose=1), ReduceLROnPlateau(monitor='loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1)], verbose=1)

threshold_output_model = Model(inputs=latent_input, outputs=[adaptive_threshold, temperature])

test_threshold, learned_temperature = threshold_output_model.predict(test_latent, verbose=0)
test_threshold = test_threshold.reshape(-1)

Y_pred = (test_error > test_threshold).astype(int)

prediction_model = Model(inputs=[latent_input, error_input], outputs=anomaly_probability)
test_probability = prediction_model.predict([test_latent, test_error.reshape(-1, 1)], verbose=0).reshape(-1)

print("Learned Temperature :", learned_temperature)
print("Adaptive Threshold Mean :", np.mean(test_threshold))
print("Adaptive Threshold Std :", np.std(test_threshold))
print("Adaptive Threshold Min :", np.min(test_threshold))
print("Adaptive Threshold Max :", np.max(test_threshold))

print("\n==============================")
print("Window Threshold Evaluation")
print("==============================")
print("Accuracy :", accuracy_score(Y_test, Y_pred))
print("Precision :", precision_score(Y_test, Y_pred, zero_division=0))
print("Recall :", recall_score(Y_test, Y_pred, zero_division=0))
print("F1 Score :", f1_score(Y_test, Y_pred, zero_division=0))
print("ROC-AUC :", roc_auc_score(Y_test, test_probability))
print(classification_report(Y_test, Y_pred, digits=4, zero_division=0))

cm = confusion_matrix(Y_test, Y_pred)

plt.figure(figsize=(7, 7))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Window Adaptive Threshold Confusion Matrix')
plt.show()

plt.figure(figsize=(14, 6))
plt.plot(test_error, label='Reconstruction Error')
plt.plot(test_threshold, label='Window Threshold')
plt.xlabel('Sequence Index')
plt.ylabel('Error / Threshold')
plt.title('Reconstruction Error vs Window Threshold')
plt.legend()
plt.grid()
plt.show()

false_positive_idx = np.where((Y_test == 0) & (Y_pred == 1))[0]
false_negative_idx = np.where((Y_test == 1) & (Y_pred == 0))[0]

print("False Positive :", len(false_positive_idx))
print("False Negative :", len(false_negative_idx))