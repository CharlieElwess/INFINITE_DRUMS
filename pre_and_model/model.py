from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import backend as k
import tensorflow as tf
from tensorflow.keras import losses
import numpy as np
from glob import glob
from tensorflow.keras.callbacks import EarlyStopping


data_dir = '/Users/felixm/Downloads/snare_specs'
audiofiles = glob(data_dir + '/*.npy')
#load
def load_data(path):
    X = []
    audiofiles = glob(data_dir + '/*.wav.npy')
    for file in audiofiles:
        X.append(np.load(file))
    return np.array(X)
data = load_data(data_dir)
#clean
def remove_nans(data):
    if np.isnan(data).sum() > 0:
        a = data.shape[1]
        b = data.shape[2]
        good_data_total = int(data.shape[0] - (np.isnan(data).sum() / (a * b)))
        data = data[~np.isnan(data)]
        data = data.reshape(good_data_total, a,b)
        return data
    return data
#t/t split
data = remove_nans(data)
X_train = data[:2000, :, :]
X_valid = data[2000:2500, :, :]
#sampling layer
class Sampling(layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        return k.random_normal(tf.shape(log_var)) * k.exp(log_var/2) + mean




codings_size = 10
#encoder
inputs = layers.Input(shape=[256,188])
z = layers.Flatten()(inputs)
z = layers.Dense(150, activation='selu')(z)
z = layers.Dense(100, activation='selu')(z)
codings_mean = layers.Dense(codings_size)(z)
codings_log_var = layers.Dense(codings_size)(z)
codings = Sampling()([codings_mean,codings_log_var])
variational_encoder = Model(
inputs=[inputs], outputs=[codings_mean, codings_log_var, codings])

#decoder
decoder_inputs = layers.Input(shape=[codings_size])
x = layers.Dense(100, activation='selu')(decoder_inputs)
x = layers.Dense(150, activation='selu')(x)
x = layers.Dense(256 * 188, activation='sigmoid')(x)
outputs = layers.Reshape([256, 188])(x)
variational_decoder = Model(inputs=[decoder_inputs], outputs=[outputs])

#combining
_,_,codings = variational_encoder(inputs)
reconstructions = variational_decoder(codings)
variational_ae = Model(inputs=[inputs], outputs=[reconstructions])

#compile
latent_loss = -0.5 * k.sum(
1 + codings_log_var - k.exp(codings_log_var) - k.square(codings_mean),
axis=-1)
variational_ae.add_loss(k.mean(latent_loss)/256*188)
variational_ae.compile(loss='binary_crossentropy', optimizer='rmsprop')

#fit
es = EarlyStopping(patience=10,restore_best_weights=True)
history = variational_ae.fit(X_train, X_train, epochs=100,
                             batch_size=128, validation_data=[X_valid, X_valid], callbacks=[es])

#latent space sampling
codings = tf.random.normal(shape=[10, codings_size],stddev=5.0)
images = variational_decoder(codings).numpy()

#save

Model.save('sus')
