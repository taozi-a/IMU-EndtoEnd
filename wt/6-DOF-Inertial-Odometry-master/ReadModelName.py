import os
import tensorflow as tf
import tensorflow.compat.v1 as tfv1
from tensorflow.python import pywrap_tensorflow



# 方法一，从python控制台看
from tensorflow.keras.models import load_model
from model import CustomMultiLossLayer

modeltrain = load_model("TRAINMODEL_epo2_bs32_adam0.0001_lossnoadj_1657282114.hdf5",
                        custom_objects={'CustomMultiLossLayer': CustomMultiLossLayer})
modeltrain.summary()
# modeltrain = load_model("TRAINMODEL_epo2_bs32_adam0.0001_lossnoadj_1657282114.hdf5")
model01 = load_model("epo1_bs32_lossnoadj_1656981897.hdf5")
model02 = load_model("6dofio_oxiod.hdf5")
# print(model)

# 方法二，没试过
"""import numpy as np
import tensorflow as tf
import os
tf.reset_default_graph()

NUM_CHANNELS=1
CONV1_DEEP=16
CONV1_SIZE=3
#名字要与训练的时候用的名字一致
conv1_weights=tf.get_variable('conv1_weights',[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP])

saver = tf.train.Saver()

MODEL_SAVE_PATH='./tensorflow_model'
MODEL_NAME='model_test.ckpt-1501'

with tf.Session() as sess:
    saver.restore(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME))
    conv1_w = conv1_weights.eval()
#    print ("weights= ", conv1_weights.eval())
    np.save('./params/conv1_w.npy',conv1_w)
"""