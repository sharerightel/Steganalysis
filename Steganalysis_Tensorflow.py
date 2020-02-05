import glob as gl
import math
import numpy as np
import tensorflow as tf
import time
import sys
import cv2
from sklearn.metrics import confusion_matrix
import time

IMAGE_SIZE = 128
NUM_CHANNELS = 1
PIXEL_DEPTH = 255.
NUM_LABELS = 2
NUM_EPOCHS = 10
Max_Steg = 50000
FLAGS = tf.app.flags.FLAGS
seed = 10
BATCH_SIZE = 10 
tf.set_random_seed(seed)

sess = tf.InteractiveSession()


def read_image(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    h = IMAGE_SIZE
    w = IMAGE_SIZE
    visual = np.zeros((h,w), np.float32)
    visual[:h, :w] = img
    return visual
            
def labeling_image(indexes):
	
    coverdirectory=FLAGS.coverdirectory
    stegdirectory=FLAGS.stegdirectory

    no = len(indexes)
    data = np.ndarray(
        shape=(no,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS),
        dtype=np.float64)
    labels = []
    for i in range(no):
        if indexes[i] < Max_Steg:
            image = read_image(coverdirectory+str(random_images[indexes[i]]+1)+".jpg")
            data[i,:,:,0] = (image/PIXEL_DEPTH)-0.5
            labels = labels + [[1.0, 0.0]]
        else:
            new_index=indexes[i]-Max_Steg
            image = read_image(stegdirectory+str(random_images[new_index]+1)+".jpg")
            data[i,:,:,0] = (image/PIXEL_DEPTH)-0.5
            labels = labels + [[0.0, 1.0]]

    labels = np.array(labels)
    return (data, labels)



def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(input=x, filter=W, strides=[1,1,1,1], padding='SAME')

#====================  Creating Convolution Layers ====================#

def create_conv_layer(in1,f_height,f_width,size_in,size_out,pooling_size,stride_size,active,fabs,paddingtype):
    # Convolution with a filter with size(f_height x f_width) 
    W_conv = weight_variable([f_height,f_width,size_in,size_out])
    z_conv=conv2d(in1, W_conv)
    if fabs==1:
        # Absolute Activation
        z_conv=tf.abs(z_conv)
    # Batch Normalization
    beta = tf.Variable(tf.constant(0.0, shape=[size_out]), name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[size_out]), name='gamma', trainable=True)
    batch_mean, batch_var = tf.nn.moments(z_conv, [0, 1, 2]  )
    ema = tf.train.ExponentialMovingAverage(decay=0.1)  #previously 0.3

    
    def update_mean_var():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
             return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(train_phase,
                        update_mean_var,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))

    BN_conv = tf.nn.batch_normalization(z_conv, mean, var, beta, gamma, epsilon)

# For TanH activation, activate should be 1, else ReLU activation will be applied
    if active==1:
        f_conv = tf.nn.tanh(BN_conv)
    else:
        f_conv = tf.nn.relu(BN_conv)
        
    # Average pooling  - pooling_size x pooling_size - stride_size - PADDING
    out = tf.nn.avg_pool(f_conv,ksize=[1,pooling_size,pooling_size,1], strides=[1,stride_size,stride_size,1], padding=paddingtype)
    return out


#====================  Creating Fully Connected Layer ====================#
#input vector: size_in components
#output vector: neurons

def my_fullcon_layer(in1,size_in,neurons):
    # Convolution with a filter with size(f_height x f_width) 
    W_full = weight_variable([size_in,neurons])
    b_full = bias_variable([neurons])
    out = tf.nn.tanh(tf.matmul(in1,W_full)+b_full)
    return out
    
#====================  Set Paths ====================#

tf.app.flags.DEFINE_string('coverdirectory', '',"""The path of cover images""")
tf.app.flags.DEFINE_string('stegdirectory', '',"""The path of stego images""")
tf.app.flags.DEFINE_string('stegodir_test', '',"""The path of test images.""")
tf.app.flags.DEFINE_string('network', '',"""The path of Pre-trained network.""")

network = FLAGS.network


# input_image is the input image  
input_image = tf.placeholder(tf.float32, shape=(BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,1))
# output_image is the expected output
output_image = tf.placeholder(tf.float32, shape=(BATCH_SIZE,2))

#====================  CNN Definition ====================#

# epsilon is used in Batch-Normalization
epsilon = 1e-4

#====================  Step1: High-pass filtering Definition ====================#

HPF = tf.cast(tf.constant([[[[-1/12.]],[[ 2/12.]], [[-2/12.]], [[2/12.]], [[-1/12.]]],[[[2/12.]],[[-6/12.]], [[8/12.]], [[-6/12.]], [[2/12.]]],[[[-2/12.]],[[8/12.]], [[-12/12.]], [[8/12.]], [[-2/12.]]],[[[2/12.]],[[-6/12.]], [[8/12.]], [[-6/12.]], [[2/12.]]],[[[-1/12.]],[[2/12.]], [[-2/12.]], [[2/12.]], [[-1/12.]]]]),"float")


#====================  Step2: The First Convolutional Layer Definition ====================#
#input: input image
#output: feature map

z_c = tf.nn.conv2d(tf.cast(input_image, "float"), HPF, strides=[1, 1, 1, 1], padding='SAME')
train_phase = tf.placeholder(tf.bool, name='train_phase')


#====================  Step3: The Second Convolutional Layer Definition ====================#
#input: 1 feature map
#output: 8 feature maps

f_conv2 = create_conv_layer(z_c,5,5,1,8,5,2,1,1,'SAME')
f_conv2_shape = f_conv2.get_shape().as_list()


#====================  Step4: The Third Convolutional Layer Definition ====================#
#input: 8 feature maps
#output: 16 feature maps

f_conv3 = create_conv_layer(f_conv2,5,5,8,16,5,2,1,0,'SAME')
f_conv3_shape = f_conv3.get_shape().as_list()


#====================  Step5: The Fourth Convolutional Layer Definition ====================#
#input: 16 feature maps
#output: 32 feature maps

f_conv4 = create_conv_layer(f_conv3,1,1,16,32,5,2,0,0,'SAME')
f_conv4_shape = f_conv4.get_shape().as_list()



#====================  Step6: The Fifth Convolutional Layer Definition ====================#
#input: 32 feature maps
#output: 64 feature maps

f_conv5 = create_conv_layer(f_conv4,1,1,32,64,5,2,0,0,'SAME')
f_conv5_shape = f_conv5.get_shape().as_list()


#====================  Step7: The Sixth Convolutional Layer Definition ====================#
#input: 64 feature maps
#output: 128 feature maps

f_conv6 = create_conv_layer(f_conv5,1,1,64,128,5,2,0,0,'SAME')
f_conv6_shape = f_conv6.get_shape().as_list()


#====================  Step8: The Seventh Convolutional Layer Definition ====================#
#input: 128 feature maps
#output: 256 feature maps

f_conv7 = create_conv_layer(f_conv6,1,1,128,256,16,1,0,0,'VALID')
f_conv7_shape = f_conv7.get_shape().as_list()


#====================  Step9: Reshaping Part for preparing the Conv layer for next layer ====================#

f_conv_shape = f_conv7.get_shape().as_list()
f_conv = tf.reshape(f_conv7,[f_conv_shape[0],f_conv_shape[1]*f_conv_shape[2]*f_conv_shape[3]])


#====================  Step10: Variables Definition ====================#

w_fc = weight_variable([256,2])
b_fc = bias_variable([2])
pred_image = tf.nn.softmax(tf.matmul(f_conv,w_fc)+b_fc)
cross_entropy = -tf.reduce_sum(output_image*tf.log(pred_image+1e-4))
train_step = tf.train.MomentumOptimizer(learning_rate=1e-3,momentum=0.9).minimize(cross_entropy)
prediction = pred_image
correct_prediction = tf.equal(tf.argmax(pred_image,1), tf.argmax(output_image,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
rounding = tf.argmax(pred_image,1)
tab = tf.placeholder(tf.float32, [None])
reduce_accuracy = tf.reduce_mean(tab)


#====================  Step11: Variables Initialization ====================#

sess.run(tf.initialize_all_variables())


#====================  Step12: Data Loading ====================#

random_images=np.arange(0,1000)
np.random.seed(seed)
np.random.shuffle(random_images)
train_image=random_images[0:500]
test_image=random_images[500:1000]


#====================  Step13: Training Data Defenition (when there is no model) ====================#

steg = np.add(train_image,np.ones(train_image.shape,dtype=np.int)*Max_Steg)
train_array = np.concatenate((train_image,steg),axis=0)

np.random.shuffle(train_array)
indexes_train = [train_array[i:i+BATCH_SIZE] for i in range(0, len(train_array), BATCH_SIZE)]
train_size = len(indexes_train)


#====================  Step14: Testing Data Defenition ====================#

steg=np.add(test_image,np.ones(test_image.shape,dtype=np.int)*Max_Steg)
test_array = np.concatenate((test_image,steg),axis=0)

np.random.seed(seed)
np.random.shuffle(test_array)
indexes_test = [test_array[i:i+BATCH_SIZE] for i in range(0, len(test_array), BATCH_SIZE)]
test_size = len(indexes_test)


#====================  Step15: Training/Loading the Network ====================#

epochsNO = NUM_EPOCHS
saver = tf.train.Saver()


#  Step15/1: Training the Network #
key=np.arange(1,3)
if network=='':
    print("Training:")
    start_time = time.time()
    for ep in range(epochsNO):
        np.random.shuffle(key)
        k_key=key[0]
        for step in range(train_size-1):
                batch_index = step 
                batch_data, batch_labels = labeling_image(indexes_train[batch_index])
                train_step.run(session=sess, feed_dict={ input_image:batch_data, output_image:batch_labels, train_phase: True })

                if step%1 == 0:
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    pred_test_index = step % test_size
                    pred_test_data, pred_test_labels = labeling_image(indexes_test[pred_test_index])
                    print("\n")
                    print("===== Step: %d, Epoch: %d, Time elapsed: %.2f ms ====="%(step,ep,1000*elapsed_time))
                    train_accuracy = accuracy.eval(session=sess, feed_dict={ input_image:batch_data, output_image:batch_labels, train_phase: True })
                    test_accuracy = accuracy.eval(session=sess, feed_dict={ input_image:pred_test_data, output_image:pred_test_labels, train_phase: False})
                    #print("Batch NO: "+str(batch_index))
                    print("Prediction - Train Accuracy: %.2f%%, Test Accuracy: %.2f%%" %(100*train_accuracy,100*test_accuracy))
                    print("loss: %.2f%%" %(100*loss))
                    
                if step == train_size-1-1:
                    global_test_predlabels = []
                    global_test_truelabels = []
                    gtest_accuracy = np.zeros(shape=(test_size), dtype=np.float32)
                    
                    train_accuracy = accuracy.eval(session=sess, feed_dict={ input_image:batch_data, output_image:batch_labels, train_phase: True })

                    for global_test_index in range(test_size-1):
                        gtest_data, gtest_labels = labeling_image(indexes_test[global_test_index])
                        batch_accuracy = accuracy.eval(session=sess, feed_dict={ input_image:gtest_data, output_image:gtest_labels, train_phase: False})
                        gtest_accuracy[global_test_index] = batch_accuracy
                        print("Global Accuracy for Batch NO. %d: %.2f%%"%(global_test_index,100*gtest_accuracy[global_test_index]))
                        gtest_predlabels = rounding.eval(session=sess, feed_dict={ input_image:gtest_data, train_phase: False})
                        global_test_predlabels = np.concatenate((global_test_predlabels,gtest_predlabels),axis=0)
                        gtest_truelabels = np.argmax(gtest_labels,1)
                        global_test_truelabels = np.concatenate((global_test_truelabels,gtest_truelabels),axis=0)
                        
                    
                    global_accuracy = reduce_accuracy.eval(session=sess, feed_dict={ tab:gtest_accuracy })
                    print("Global Accuracy for Test:%.2f%%"%(100*global_accuracy))
                    print("Matrix Confusion:")
                    print(confusion_matrix(global_test_predlabels,global_test_truelabels))

                    np.random.shuffle(train_array)
                    indexes_train = [train_array[i:i+BATCH_SIZE] for i in range(0, len(train_array), BATCH_SIZE)]
                    train_size = len(indexes_train)
                    print("Shuffle")
                    
                    #It is possible to save the model for future
                    #saver.save(sess, "network", global_step=ep+1) 
                    
                    
                    
#  Step15/2: Loading a Network #
else:
    print("Loading a Network:")
    saver.restore(sess, network)

    global_test_predlabels = []
    global_test_truelabels = []
    gtest_accuracy = np.ndarray(shape=(test_size), dtype=np.float32)
    for global_test_index in range(test_size-1):
        gtest_data, gtest_labels = labeling_image(indexes_test[global_test_index])
        batch_accuracy = accuracy.eval(session=sess, feed_dict={ x:gtest_data, y:gtest_labels, train_phase.name: False})
        gtest_accuracy[global_test_index] = batch_accuracy
        print("Global accuracy for batch NO %d: %.2f%%"%(global_test_index,100*gtest_accuracy[global_test_index]))
        gtest_predlabels = rounding.eval(session=sess, feed_dict={ x:gtest_data, train_phase.name: False})
        global_test_predlabels = np.concatenate((global_test_predlabels,gtest_predlabels),axis=0)
        gtest_truelabels = np.argmax(gtest_labels,1)
        global_test_truelabels = np.concatenate((global_test_truelabels,gtest_truelabels),axis=0)
        
    global_accuracy = reduce_accuracy.eval(session=sess, feed_dict={ tab:gtest_accuracy })
    print("Global Accuracy for Test: %.2f%%"%(100*global_accuracy))
    print("Matrix Confusion:")
    print(confusion_matrix(global_test_predlabels,global_test_truelabels))