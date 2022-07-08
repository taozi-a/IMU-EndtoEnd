import tfquaternion as tfq

# from tensorflow.compat.v1.keras.layers import CuDNNLSTM as LSTM
import tensorflow.compat.v1
# from tensorflow.keras import Sequential, Model
# from tensorflow.keras.layers import Bidirectional, LSTM, Input, concatenate, MaxPooling1D
# from tensorflow.keras.layers import Bidirectional, Dropout, Dense, Input, Layer, Conv1D, MaxPooling1D, concatenate
# from tensorflow.keras.layers import Bidirectional, LSTM, CuDNNLSTM, Dropout, Dense, Input, Layer, Conv1D, MaxPooling1D, concatenate
from tensorflow.compat.v1 import keras
# from tensorflow.compat.v1.keras.layers import CuDNNLSTM
# from tensorflow.keras.layers import CuDNNLSTM
# from tensorflow.keras.initializers import Constant
from tensorflow.compat.v1.initializers import constant
# import tensorflow.compat.v1.keras.initializers.constant
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.losses import mean_absolute_error
# from tensorflow.keras import backend as K
# from tensorflow.compat.v1.layers import Dropout, Dense, Layer, Conv1D

import tensorflow as tf


def quaternion_phi_3_error(y_true, y_pred):
    return tf.compat.v1.acos(tf.compat.v1.keras.backend.abs(
        tf.compat.v1.keras.backend.batch_dot(y_true, tf.compat.v1.keras.backend.l2_normalize(y_pred, axis=-1),
                                             axes=-1)))


def quaternion_phi_4_error(y_true, y_pred):
    return 1 - tf.compat.v1.keras.backend.abs(
        tf.compat.v1.keras.backend.batch_dot(y_true, tf.compat.v1.keras.backend.l2_normalize(y_pred, axis=-1), axes=-1))


def quaternion_log_phi_4_error(y_true, y_pred):
    return tf.compat.v1.keras.backend.log(1e-4 + quaternion_phi_4_error(y_true, y_pred))


def quat_mult_error(y_true, y_pred):
    q_hat = tfq.Quaternion(y_true)
    q = tfq.Quaternion(y_pred).normalized()  # 预测值归一化后的tfq类型四元数
    q_prod = q * (q_hat.conjugate())  # """Compute the conjugate of q, i.e. [q.w, -q.x, -q.y, -q.z]."""
    w, x, y, z = tf.compat.v1.split(q_prod, num_or_size_splits=4, axis=-1)
    return tf.compat.v1.abs(tf.compat.v1.multiply(2.0, tf.compat.v1.concat(values=[x, y, z], axis=-1)))


def quaternion_mean_multiplicative_error(y_true, y_pred):
    # return tf.reduce_mean(quatmult_error(y_true, y_pred))
    return tf.compat.v1.reduce_mean(quat_mult_error(y_true, y_pred))


# Custom loss layer
class CustomMultiLossLayer(keras.layers.Layer):
    def __init__(self, nb_outputs=2, **kwargs):
        # def __init__(self, nb_outputs=3, **kwargs):
        self.nb_outputs = nb_outputs
        # self.is_placeholder = True
        super(CustomMultiLossLayer, self).__init__(**kwargs)

    def build(self, input_shape=None):
        # initialise log_vars
        self.log_vars = []
        for i in range(self.nb_outputs):
            # self.log_vars += [self.add_weight(name='log_var' + str(i), shape=(1,),
            #                                   initializer=constant(0.), trainable=True)]
            self.log_vars += [self.add_weight(name='log_var' + str(i), shape=(1,),
                                              initializer="random_normal", trainable=True)]

        super(CustomMultiLossLayer, self).build(input_shape)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'nb_outputs': self.nb_outputs,
            # 'is_placeholder': self.is_placeholder
        })
        return config

    def multi_loss(self, ys_true, ys_pred):
        assert len(ys_true) == self.nb_outputs and len(ys_pred) == self.nb_outputs
        loss = 0

        # for y_true, y_pred, log_var in zip(ys_true, ys_pred, self.log_vars):
        #    precision = K.exp(-log_var[0])
        #    loss += K.sum(precision * (y_true - y_pred)**2., -1) + log_var[0]

        precision = tf.compat.v1.keras.backend.exp(-self.log_vars[0][0])
        loss += precision * tensorflow.compat.v1.keras.losses.mean_absolute_error(ys_true[0], ys_pred[0]) + \
                self.log_vars[0][0]
        precision = tf.compat.v1.keras.backend.exp(-self.log_vars[1][0])
        loss += precision * quaternion_mean_multiplicative_error(ys_true[1], ys_pred[1]) + self.log_vars[1][0]
        # loss = tensorflow.compat.v1.abs(loss)
        # loss += precision * quaternion_phi_4_error(ys_true[1], ys_pred[1]) + self.log_vars[1][0]
        # print(tf.compat.v1.keras.backend.mean(loss).shape)
        return tf.compat.v1.keras.backend.mean(loss)

    def call(self, inputs):
        # We won't actually use the output.
        ys_true = inputs[:self.nb_outputs]
        ys_pred = inputs[self.nb_outputs:]
        loss = self.multi_loss(ys_true, ys_pred)
        # print(loss)
        # print(type(loss), loss.shape)
        # print(type(tf.compat.v1.keras.backend.concatenate(inputs, -1)),
        #       tf.compat.v1.keras.backend.concatenate(inputs, -1).shape)
        self.add_loss(loss, inputs=inputs)
        return tf.compat.v1.keras.backend.concatenate(inputs, -1)
        # return tf.compat.v1.keras.backend.concatenate(loss, -1)
        # return loss

    def compute_output_shape(self, inputs):
        ys_true = inputs[:self.nb_outputs]
        ys_pred = inputs[self.nb_outputs:]
        loss = self.multi_loss(ys_true, ys_pred)
        return loss


def create_pred_model_6d_quat(window_size=200):
    # inp = Input((window_size, 6), name='inp')
    # lstm1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(inp)
    x1 = keras.layers.Input((window_size, 3), name='x1')
    x2 = keras.layers.Input((window_size, 3), name='x2')
    convA1 = keras.layers.Conv1D(128, 11)(x1)
    convA2 = keras.layers.Conv1D(128, 11)(convA1)
    poolA = keras.layers.MaxPooling1D(3, strides=1)(convA2)

    convB1 = keras.layers.Conv1D(128, 11)(x2)
    convB2 = keras.layers.Conv1D(128, 11)(convB1)
    poolB = keras.layers.MaxPooling1D(3, strides=1)(convB2)
    AB = keras.layers.concatenate([poolA, poolB])
    # lstm1 = Bidirectional(LSTM(128, return_sequences=True))(AB)
    # lstm1 = Bidirectional(tf.keras.layers.CuDNNLSTM(128, return_sequences=True))(AB)
    lstm1 = keras.layers.Bidirectional(tensorflow.compat.v1.keras.layers.CuDNNLSTM(128, return_sequences=True))(AB)
    drop1 = keras.layers.Dropout(0.25)(lstm1)
    # lstm2 = Bidirectional(LSTM(128))(drop1)
    # lstm2 = Bidirectional(tf.keras.layers.CuDNNLSTM(128))(drop1)
    # sigmoid1 = keras.activations.sigmoid(drop1)
    # GRU = keras.layers.CuDNNGRU(128)(drop1)
    # ddop1 = tf.reshape(drop1[1], [4, 32, -1])
    # lstm3 = keras.layers.Bidirectional(tensorflow.compat.v1.keras.layers.CuDNNGRU(64))(ddop1)
    # drop3 = keras.layers.Dropout(0.25)(lstm3)
    # ddop2 = tf.reshape(drop3, [1, 1, -1])
    convC = keras.layers.Conv1D(128, 11)(drop1)
    poolC = keras.layers.MaxPooling1D(3, strides=1)(convC)
    lstm2 = keras.layers.Bidirectional(tensorflow.compat.v1.keras.layers.CuDNNLSTM(128))(convC)
    drop2 = keras.layers.Dropout(0.25)(lstm2)

    # newshape = tensorflow.convert_to_tensor([2, 2, -1])
    # drop2 = tensorflow.reshape(drop2[1], newshape)
    #
    # lstm3 = keras.layers.CuDNNLSTM(64)(drop2)
    # drop3 = keras.layers.Dropout(0.25)(lstm3)

    y1_pred = keras.layers.Dense(3)(drop2)
    y2_pred = keras.layers.Dense(4)(drop2)

    # model = Model(inp, [y1_pred, y2_pred])

    model = tensorflow.compat.v1.keras.Model([x1, x2], [y1_pred, y2_pred])
    model.summary()
    return model


def create_train_model_6d_quat(pred_model, window_size=200):
    # inp = Input(shape=(window_size, 6), name='inp')
    # y1_pred, y2_pred = pred_model(inp)
    x1 = keras.layers.Input(shape=(window_size, 3), name='x1')
    x2 = keras.layers.Input(shape=(window_size, 3), name='x2')
    y1_pred, y2_pred = pred_model([x1, x2])
    y1_true = keras.layers.Input(shape=(3,), name='y1_true')
    y2_true = keras.layers.Input(shape=(4,), name='y2_true')
    out = CustomMultiLossLayer(nb_outputs=2)([y1_true, y2_true, y1_pred, y2_pred])
    # train_model = Model([inp, y1_true, y2_true], out)
    train_model = tensorflow.compat.v1.keras.Model([x1, x2, y1_true, y2_true], out)
    train_model.summary()
    return train_model


# def create_predict_train_model(pred_model, window_size=200):
#     # inp = Input(shape=(window_size, 6), name='inp')
#     # y1_pred, y2_pred = pred_model(inp)
#     x1 = keras.layers.Input(shape=(window_size, 3), name='x1')
#     x2 = keras.layers.Input(shape=(window_size, 3), name='x2')
#     y1_pred, y2_pred = pred_model([x1, x2])
#     y1_true = keras.layers.Input(shape=(3,), name='y1_true')
#     y2_true = keras.layers.Input(shape=(4,), name='y2_true')
#     out = CustomMultiLossLayer(nb_outputs=2).multi_loss([y1_true, y2_true, y1_pred, y2_pred])
#     # train_model = Model([inp, y1_true, y2_true], out)
#     train_model = tensorflow.compat.v1.keras.Model([x1, x2, y1_true, y2_true], out)
#     train_model.summary()
#     return train_model


def create_pred_model_3d(window_size=200):
    # inp = Input((window_size, 6), name='inp')
    # lstm1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(inp)
    x1 = keras.layers.Input((window_size, 3), name='x1')
    x2 = keras.layers.Input((window_size, 3), name='x2')
    convA1 = keras.layers.Conv1D(128, 11)(x1)
    convA2 = keras.layers.Conv1D(128, 11)(convA1)
    poolA = keras.layers.MaxPooling1D(3)(convA2)
    convB1 = keras.layers.Conv1D(128, 11)(x2)
    convB2 = keras.layers.Conv1D(128, 11)(convB1)
    poolB = keras.layers.MaxPooling1D(3)(convB2)
    AB = keras.layers.concatenate([poolA, poolB])
    # lstm1 = Bidirectional(CCuDNNLSTM128, return_sequences=True))(AB)
    lstm1 = keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(AB)
    drop1 = keras.layers.Dropout(0.25)(lstm1)
    # lstm2 = Bidirectional(CuDNNLSTM(128))(drop1)
    lstm2 = keras.layers.Bidirectional(tf.keras.layers.LSTM(128))(drop1)
    drop2 = keras.layers.Dropout(0.25)(lstm2)
    y1_pred = keras.layers.Dense(1)(drop2)
    y2_pred = keras.layers.Dense(1)(drop2)
    y3_pred = keras.layers.Dense(1)(drop2)

    # model = Model(inp, [y1_pred, y2_pred, y3_pred])
    model = tensorflow.compat.v1.keras.Model([x1, x2], [y1_pred, y2_pred, y3_pred])

    model.summary()

    return model


def create_train_model_3d(pred_model, window_size=200):
    # inp = Input(shape=(window_size, 6), name='inp')
    # y1_pred, y2_pred, y3_pred = pred_model(inp)
    x1 = keras.layers.Input((window_size, 3), name='x1')
    x2 = keras.layers.Input((window_size, 3), name='x2')
    y1_pred, y2_pred, y3_pred = pred_model([x1, x2])
    y1_true = keras.layers.Input(shape=(1,), name='y1_true')
    y2_true = keras.layers.Input(shape=(1,), name='y2_true')
    y3_true = keras.layers.Input(shape=(1,), name='y3_true')
    out = CustomMultiLossLayer(nb_outputs=3)([y1_true, y2_true, y3_true, y1_pred, y2_pred, y3_pred])
    # train_model = Model([inp, y1_true, y2_true, y3_true], out)
    train_model = tensorflow.compat.v1.keras.Model([x1, x2, y1_true, y2_true, y3_true], out)
    train_model.summary()
    return train_model


def create_model_6d_rvec(window_size=200):
    input_gyro_acc = keras.layers.Input((window_size, 6))
    # lstm1 = Bidirectional(LSTM(128, return_sequences=True))(input_gyro_acc)
    lstm1 = keras.layers.Bidirectional(tensorflow.compat.v1.keras.layers.CuDNNLSTM(128, return_sequences=True))(
        input_gyro_acc)
    drop1 = keras.layers.Dropout(0.25)(lstm1)
    # lstm2 = Bidirectional(LSTM(128))(drop1)
    lstm2 = keras.layers.Bidirectional(tensorflow.compat.v1.keras.layers.CuDNNLSTMM(128))(drop1)
    drop2 = keras.layers.Dropout(0.25)(lstm2)
    output_delta_rvec = keras.layers.Dense(3)(drop2)
    output_delta_tvec = keras.layers.Dense(3)(drop2)

    model = tensorflow.compat.v1.keras.Model(inputs=input_gyro_acc, outputs=[output_delta_rvec, output_delta_tvec])
    model.summary()
    model.compile(optimizer=tensorflow.compat.v1.keras.optimizers.Adam(0.0001), loss='mean_squared_error')

    return model


def create_model_6d_quat(window_size=200):
    input_gyro_acc = keras.layers.Input((window_size, 6))
    # lstm1 = Bidirectional(LSTM(128, return_sequences=True))(input_gyro_acc)
    lstm1 = keras.layers.Bidirectional(tensorflow.compat.v1.keras.layers.CuDNNLSTM(128, return_sequences=True))(
        input_gyro_acc)
    drop1 = keras.layers.Dropout(0.25)(lstm1)
    # lstm2 = Bidirectional(LSTM(128))(drop1)
    lstm2 = keras.layers.Bidirectional(tensorflow.compat.v1.keras.layers.CuDNNLSTM(128))(drop1)
    drop2 = keras.layers.Dropout(0.25)(lstm2)
    output_delta_p = keras.layers.Dense(3)(drop2)
    output_delta_q = keras.layers.Dense(4)(drop2)

    model = tensorflow.compat.v1.keras.Model(inputs=input_gyro_acc, outputs=[output_delta_p, output_delta_q])
    model.summary()
    # model.compile(optimizer = Adam(0.0001), loss = 'mean_squared_error')
    model.compile(optimizer=tensorflow.compat.v1.keras.optimizers.Adam(0.0001),
                  loss=['mean_absolute_error', quaternion_mean_multiplicative_error])
    # model.compile(optimizer = Adam(0.0001), loss = ['mean_absolute_error', quaternion_phi_4_error])

    return model


def create_model_3d(window_size=200):
    input_gyro_acc = keras.layers.Input((window_size, 6))
    # lstm1 = Bidirectional(LSTM(128, return_sequences=True))(input_gyro_acc)
    lstm1 = keras.layers.Bidirectional(tensorflow.compat.v1.keras.layers.CuDNNLSTM(128, return_sequences=True))(
        input_gyro_acc)
    # tensorflow.compat.v1.keras.layers.CuDNNLSTM
    drop1 = keras.layers.Dropout(0.25)(lstm1)
    # lstm2 = Bidirectional(LSTM(128))(drop1)
    lstm2 = keras.layers.Bidirectional(tensorflow.compat.v1.keras.layers.CuDNNLSTM(128))(drop1)
    drop2 = keras.layers.Dropout(0.25)(lstm2)
    output_delta_l = keras.layers.Dense(1)(drop2)
    output_delta_theta = keras.layers.Dense(1)(drop2)
    output_delta_psi = keras.layers.Dense(1)(drop2)

    model = tensorflow.compat.v1.keras.Model(inputs=input_gyro_acc,
                                             outputs=[output_delta_l, output_delta_theta, output_delta_psi])
    model.summary()
    model.compile(optimizer=tensorflow.compat.v1.keras.optimizers.Adam(0.0001), loss='mean_squared_error')

    return model


def create_model_2d(window_size=200):
    input_gyro_acc = keras.layers.Input((window_size, 6))
    # lstm1 = Bidirectional(LSTM(128, return_sequences=True))(input_gyro_acc)
    lstm1 = keras.layers.Bidirectional(tensorflow.compat.v1.keras.layers.CuDNNLSTM(128, return_sequences=True))(
        input_gyro_acc)
    drop1 = keras.layers.Dropout(0.25)(lstm1)
    # lstm2 = Bidirectional(LSTM(128))(drop1)
    lstm2 = keras.layers.Bidirectional(tensorflow.compat.v1.keras.layers.CuDNNLSTM(128))(drop1)
    drop2 = keras.layers.Dropout(0.25)(lstm2)
    output_delta_l = keras.layers.Dense(1)(drop2)
    output_delta_psi = keras.layers.Dense(1)(drop2)
    model = tensorflow.compat.v1.keras.Model(inputs=input_gyro_acc, outputs=[output_delta_l, output_delta_psi])
    model.summary()
    model.compile(optimizer=tensorflow.compat.v1.keras.optimizers.Adam(0.0001), loss='mean_squared_error')

    return model
