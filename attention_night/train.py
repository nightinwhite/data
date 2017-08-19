#coding:utf-8
import tensorflow as tf
import tensorflow.contrib.slim as slim
from read_tfrecord import *
import cv2
import numpy as np
import sys
import model
import os
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import LSTMCell
import common_flag
from tensorflow.python.platform import flags
from pprint import pprint
import tensorflow.contrib.deprecated as tfsummary
FLAGS = flags.FLAGS
common_flag.define()
#model parameter
batch_size = 32
IMG_WIDTH = FLAGS.img_width
IMG_HEIGHT = FLAGS.img_height
IMG_CHANNEL = FLAGS.img_channel
WEIGHT_DECAY = FLAGS.weight_decay
num_classes = FLAGS.num_char_classes
epoch = 10000
learning_rate = FLAGS.learning_rate
iteration_in_epoch = 1000
iteration_in_test = 100


#--------------------------------------------------------------------
#data
net_input = tf.placeholder("float",[batch_size,FLAGS.img_height,FLAGS.img_width,FLAGS.img_channel])
net_image_label = tf.placeholder("float",[batch_size,FLAGS.seq_length,FLAGS.num_char_classes])
net_learning = tf.placeholder("float",None)
#model
attention_model = model.Attention_Model(net_input,net_image_label)
attention_model.create_model()
#img_loss
img_loss = attention_model.loss
net_predict = attention_model.predict
net_accuracy = tf.cast(tf.equal(net_predict,tf.arg_max(net_image_label,-1)),tf.float32)
net_acc = tf.reduce_sum(net_accuracy, -1)
net_accuracy = tf.reduce_mean(net_accuracy)
net_acc = tf.cast(tf.equal(net_acc, FLAGS.seq_length), tf.float32)
net_acc = tf.reduce_mean(net_acc)
# ---------------------------------------------
#all loss
net_loss = img_loss
net_optimizer = tf.train.MomentumOptimizer(FLAGS.learning_rate,FLAGS.momentum)
net_train = slim.learning.create_train_op(net_loss,net_optimizer,summarize_gradients=True,clip_gradient_norm=FLAGS.clip_gradient_norm)
# tst_loss_l = slim.losses.get_regularization_losses()
# tst_loss = tst_loss_l[0]
# for i in range(1,len(tst_loss_l)):
#     tst_loss+=tst_loss_l[i]

#initialize weight
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()
# summarys:
all_variables = tf.global_variables()
for v in all_variables:
    tf.summary.histogram(v.name,v)
tf.summary.image("train_images",net_input,10)
tf.summary.scalar("net_loss",net_loss)
tf.summary.scalar("net_accuracy",net_accuracy)
tf.summary.scalar("net_acc",net_acc)
merge_summary = tfsummary.merge_all_summaries()
summary_writer = tf.summary.FileWriter("train_log/", sess.graph)
saver.restore(sess, "model/attention_bac_64_10.data")

#load data
data_train_img,data_train_labels = net_read_data(["tfrecords/train_bac_64.records"],
                                    num_classes,IMG_WIDTH,IMG_HEIGHT,IMG_CHANNEL)
batch_train_imgs,batch_train_labels = tf.train.shuffle_batch([data_train_img,data_train_labels],batch_size,1000,100,4)

data_test_img,data_test_labels = net_read_data(["tfrecords/test_bac_64.records"],
                                    num_classes,IMG_WIDTH,IMG_HEIGHT,IMG_CHANNEL,is_train_set=False)
batch_test_imgs,batch_test_labels = tf.train.shuffle_batch([data_test_img,data_test_labels],batch_size,1000,100,4)

theads = tf.train.start_queue_runners(sess)
#test_net
# tst = net_output
# res = sess.run(tst,feed_dict={net_input : test_imgs})
# res = np.asarray(res)
# print res.shape
# print res

#train model
best_rate = 0.90
for e in range(epoch):
    e_loss = 0
    e_acc = 0
    e_all_acc = 0
    for i in range(iteration_in_epoch):
        gen_imgs,gen_labels = sess.run([batch_train_imgs,batch_train_labels])
        # pprint (gen_imgs.shape)
        # pprint( gen_labels.shape)
        # gen_label = gen_labels[1]
        # gen_label = np.argmax(gen_label,1)
        # pprint(gen_label)
        # gen_min = np.min(gen_imgs[1])
        # gen_max = np.max(gen_imgs[1])
        # gen_img = (gen_imgs[1] - gen_min)/(gen_max-gen_min)*255
        # show_img = np.asarray(gen_img,np.uint8)
        # cv2.imshow("0",show_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # pprint( gen_imgs[0])
        # print ""
        # pprint(gen_labels[0])
        # print ""
        # tst = np.asarray(gen_labels[0])
        # pprint( np.argmax(tst,-1))
        # pprint( np.max(tst))
        i_loss,i_acc,i_all_acc,_ = sess.run([net_loss,net_accuracy,net_acc,net_train],feed_dict={
            net_input : gen_imgs,net_image_label: gen_labels,net_learning:learning_rate
        })
        e_loss += i_loss
        e_acc += i_acc
        e_all_acc += i_all_acc
        # if i_acc>e_tst:
        #     e_tst = i_acc
        # f = open("tst_info", "a")
        # f.write("epoch:{0}/{1},iter:{2}/{3}:\nit_loss:{4}\n".format(
        #     e + 1, epoch, i + 1, iteration_in_epoch, i_acc
        # ))
        # f.close()
        sys.stdout.write("                                                                                \r")
        sys.stdout.flush()
        sys.stdout.write("epoch:{0}/{1},iter:{2}/{3},loss:{4},acc:{5},all_acc:{6}".format(
            e+1,epoch,i+1,iteration_in_epoch,e_loss/(i+1.),e_acc/(i+1.),e_all_acc/(i+1.)
        ))
    print ""
    t_loss = 0
    t_acc = 0
    t_all_acc = 0
    for i in range(iteration_in_test):
        gen_imgs,gen_labels = sess.run([batch_test_imgs,batch_test_labels])
        i_loss,i_acc,i_all_acc = sess.run([net_loss,net_accuracy,net_acc],feed_dict={
            net_input : gen_imgs,net_image_label: gen_labels,net_learning:learning_rate
        })
        t_loss += i_loss
        t_acc += i_acc
        t_all_acc += i_all_acc

    print "epoch:{0}/{1},loss:{2},acc:{3},acc_all:{4}".format(
            e+1,epoch,t_loss/(iteration_in_test),t_acc/(iteration_in_test),t_all_acc/(iteration_in_test)
        )
    if t_all_acc/(iteration_in_test) > best_rate:
    	saver.save(sess, "model/attention_bac_64_{0}_{1}.data".format(e,t_all_acc/(iteration_in_test)))
	best_rate = t_all_acc/(iteration_in_test)
