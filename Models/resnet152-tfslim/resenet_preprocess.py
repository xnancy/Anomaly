import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import os

tf.reset_default_graph()

resnet1 = nets.resnet_v1

# Specify where the pretrained Model is saved.
model_path = 'resnet_v1_152.ckpt'

# Specify where the new model will live
log_dir = 'resnet_log/'

tf.reset_default_graph()

images = tf.placeholder(tf.float32, [None, 224, 224, 3])

# with slim.arg_scope(resnet2.resnet_arg_scope()):
net, end_points = resnet1.resnet_v1_152(images, 1001)

assert(os.path.isfile(model_path)) 

variables_to_restore = tf.contrib.framework.get_variables_to_restore()

restorer = tf.train.Saver(variables_to_restore)


init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    restorer.restore(sess,model_path)
    print(session.run(tf.all_variables()))
    print("model restored!")
    

 
