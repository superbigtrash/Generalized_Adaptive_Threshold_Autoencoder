import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras import Model, layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix

# 0. Seed
np.random.seed(42)
tf.random.set_seed(42)

# 1. Positional Encoding
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

def flatten(X):
    flattened = np.empty((X.shape[0], X.shape[2]))
    for i in range(X.shape[0]):
        flattened[i] = X[i,X.shape[1] - 1,:]
    return(flattened)

# 2. Transformer Encoder Block
def transformer_encoder(x, d_model, num_heads, ff_dim, dropout=0.1):

    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(query=x, value=x, key=x)
    attention_output = layers.Dropout(dropout)(attention_output) # multi-head attention dropout

    x = layers.LayerNormalization(epsilon=1e-6)(x + attention_output) # Add & Norm

    ffn_output = layers.Dense(ff_dim,activation='relu')(x)
    ffn_output = layers.Dense(d_model)(ffn_output)
    ffn_output = layers.Dropout(dropout)(ffn_output) # Feed Forward Network dropout

    x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output) # Add & Norm
    return x

# 3. Transformer Autoencoder
def build_transformer_autoencoder(
    sequence_length=50,
    n_features=3,
    d_model=64,
    num_heads=4,
    ff_dim=128,
    latent_dim=16,
    num_encoder_blocks=2,
    dropout=0.1):

    inputs = layers.Input(shape=(sequence_length, n_features), name='input_sequence')

    # Input Embedding (sequence, 3)->(sequence, d_model)
    x = layers.Dense(d_model, name='input_embedding')(inputs)
    x = PositionalEncoding(sequence_length=sequence_length, d_model=d_model)(x) # Positional Encoding

    for _ in range(num_encoder_blocks): # Transformer Encoder
        x = transformer_encoder(x=x, d_model=d_model, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout)
        
    x = layers.GlobalAveragePooling1D(name='global_average_pooling')(x)
    x = layers.Dense(64, activation='relu', name='encoder_dense')(x)
    latent = layers.Dense(latent_dim, activation=None, name='latent_vector')(x) # Bottleneck

    x = layers.Dense(sequence_length * d_model,activation='relu', name='decoder_dense')(latent)
    x = layers.Reshape((sequence_length, d_model), name='decoder_reshape')(x)
    x = layers.Dense(64, activation='relu', name='decoder_ffn')(x) # Decoder

    outputs = layers.Dense(n_features, activation=None, name='reconstruction')(x)
    model = Model(inputs=inputs, outputs=outputs, name='Transformer_Autoencoder') # Reconstruction
    return model

# 4. Sequence 생성 함수
def make_sequence(data, labels, sequence_length):
    X, Y = [], []

    for i in range(len(data) - sequence_length + 1):
        X.append(data[i:i + sequence_length])

        # window 마지막 시점의 label
        Y.append(labels[i + sequence_length - 1])
    return (np.array(X), np.array(Y))

# 5. Dataset Load
normal = pd.read_csv('./press_data_normal.csv', index_col=0) 
outlier = pd.read_csv('./outlier_data.csv', index_col=0) 

normal_data = normal.copy()
outlier_data = outlier.copy()

use_col = ['AI0_Vibration', 'AI1_Vibration', 'AI2_Current']

X_normal = normal_data[use_col]
y_normal = normal_data['Equipment_state']

X_anomaly = outlier_data[use_col]
y_anomaly = outlier_data['Equipment_state']

print("Normal shape :", X_normal.shape)
print("Anomaly shape :", X_anomaly.shape)

# 7. Train / Test 분리
X_train_normal = X_normal[:15000]
y_train_normal = y_normal[:15000]

X_test_normal = X_normal[15000:]
y_test_normal = y_normal[15000:]

X_test_anomaly = X_anomaly
y_test_anomaly = y_anomaly

# 8. Scaling 정상 Train 데이터로만 fit
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_normal)
X_test_normal_scaled = scaler.transform(X_test_normal)
X_test_anomaly_scaled = scaler.transform(X_test_anomaly)

y_train_normal = np.array(y_train_normal)
y_test_normal = np.array(y_test_normal)
y_test_anomaly = np.array(y_test_anomaly)

# 9. Sequence 생성  가이드북 LSTM-AE와 동일
SEQUENCE_LENGTH = 50
FUTURE_OFFSET = 100
TH_LEARNING_RATE = 0.01
TF_LEARNING_RATE = 0.001
TF_EPOCHS = 100
THRESHOLD_EPOCHS = 600
TF_BATCH_SIZE = 64

N_FEATURES = 3
D_MODEL = 64
NUM_HEADS = 4
FF_DIM = 128
LATENT_DIM = 32
NUM_ENCODER_BLOCKS = 2
DROPOUT = 0.1

NORMAL_WEIGHT = 1.0
ANOMALY_WEIGHT = 20.0

SMOOTH_WEIGHT = 100.0
THRESHOLD_WEIGHT = 4.0

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

# 10. Validation / Test 분리    가이드북과 동일
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

# 13. Model 생성
model = build_transformer_autoencoder(
    sequence_length=SEQUENCE_LENGTH,
    n_features=N_FEATURES,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    ff_dim=FF_DIM,
    latent_dim=LATENT_DIM,
    num_encoder_blocks=NUM_ENCODER_BLOCKS, 
    dropout=DROPOUT)

# 14. Compile
model.compile(optimizer=Adam(learning_rate=TF_LEARNING_RATE),loss='mse')
model.summary()

# 15. Callback
early_stopping = EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True,verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=5,min_lr=1e-6,verbose=1)


# 16. Train  Autoencoder: X_train -> X_train 정상 데이터만 학습
history = model.fit(X_train,X_train,
                    validation_data=(X_valid_0, X_valid_0),
                    epochs=TF_EPOCHS, batch_size=TF_BATCH_SIZE, shuffle=False,
                    callbacks=[early_stopping,reduce_lr], verbose=1)

model.save_weights('./weights/tf_ae.weights.h5')

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

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='AE Train Loss')
plt.plot(history.history['val_loss'], label='AE Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Transformer Autoencoder Loss')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(threshold_history.history['loss'], label='Threshold Total Loss')
plt.plot(threshold_history.history['classification_loss'], label='Classification Loss')
plt.plot(threshold_history.history['smooth_loss'], label='Smooth Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Adaptive Threshold Loss')
plt.legend()
plt.grid()
plt.show()