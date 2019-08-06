
### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
from random import randint
print(randint(0,9))
# Visualizations will be shown in the notebook.
%matplotlib inline

#Display 3 random images within the datasets.
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
f.tight_layout()
#Training set
rand_train = randint(0, (n_train)-1)
ax1.imshow(X_train[rand_train])
ax1.set_title("Training Image", fontsize=25)

#Testing set
rand_test = randint(0, (n_test)-1)
ax2.imshow(X_test[rand_test])
ax2.set_title("Testing Image", fontsize=25)

#Validation set
rand_valid = randint(0, (n_validation)-1)
ax3.imshow(X_valid[rand_valid])
ax3.set_title("Validation Image", fontsize=25)

plt.figure(figsize=(24, 9))
plt.hist(y_train, bins=n_classes)
plt.hist(y_test, bins=n_classes)
plt.xlabel('Sign ID')
plt.ylabel('Count of each sign')
plt.show()


##########################################################################################################################################

##########################################################################################################################################

#Preprocess the data here. 
import cv2 as cv
import numpy as np

#Empty arrays to store temp normalized images
temp_Xtrain = []
temp_Xtest = []
temp_Xvalid = []

def norm(image):
    img = image
    
    #Equalizing the Y-channel and combining it with the other two channels to get the output image
    img_hls = cv.cvtColor(img, cv.COLOR_BGR2HLS)
    img_hls[:,:,1] = cv.equalizeHist(img_hls[:,:,1])
    img_rgb = cv.cvtColor(img_hls, cv.COLOR_HLS2RGB)

    img_hsv = cv.cvtColor(img_rgb, cv.COLOR_RGB2HSV)

    
    img_hsv[:,:,2] = cv.equalizeHist(img_hsv[:,:,2])
    img_out = cv.cvtColor(img_hsv, cv.COLOR_HSV2RGB)
    
    #Get pixels
    r = img_out[:,:,0]
    g = img_out[:,:,1]
    b = img_out[:,:,2]
    
    #Approxamitly normalize data
    blue = (b - 128. )/ 128.
    green = (g - 128.) / 128.
    red = (r - 128.) / 128.
    
    pixels = cv.merge((blue,green,red))
    
    return pixels

#Normalize training data
for i in range(len(X_train)):
    temp_Xtrain.append(norm(X_train[i]))
X_train = np.array(temp_Xtrain)
print("X_train normalization done...")

#Normalize testing data
for i in range(len(X_test)):
    temp_Xtest.append(norm(X_test[i]))
X_test = np.array(temp_Xtest)
print("X_test normalization done...")

#Normalize validation data
for i in range(len(X_valid)):
    temp_Xvalid.append(norm(X_valid[i]))
X_valid = np.array(temp_Xvalid)
print("X_valid normalization done!")

#Display 3 normailized images.
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
f.tight_layout()
#Training set
rand_train = randint(0, (n_train)-1)
ax1.imshow(X_train[rand_train])
ax1.set_title("Normalized Training Image", fontsize=25)

#Testing set
rand_test = randint(0, (n_test)-1)
ax2.imshow(X_test[rand_test])
ax2.set_title("Normalized Testing Image", fontsize=25)

#Validation set
rand_valid = randint(0, (n_validation)-1)
ax3.imshow(X_valid[rand_valid])
ax3.set_title("Normalized Validation Image", fontsize=25)

from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train, y_train)

import tensorflow as tf

EPOCHS = 30
BATCH_SIZE = 64

##########################################################################################################################################

##########################################################################################################################################

#Define model architecture.
from tensorflow.contrib.layers import flatten

def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    #Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    conv_1_w = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma))
    conv_1_b = tf.Variable(tf.zeros(6))
    conv_1  = tf.nn.conv2d(x, conv_1_w, strides=[1, 1, 1, 1], padding='VALID') + conv_1_b

    #Activation.
    conv_1 = tf.nn.relu(conv_1)

    #Pooling. Input = 28x28x6. Output = 14x14x6.
    conv_1 = tf.nn.max_pool(conv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    #Layer 2: Convolutional. Output = 10x10x16.
    conv_2_w = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
    conv_2_b = tf.Variable(tf.zeros(16))
    conv_2 = tf.nn.conv2d(conv_1, conv_2_w, strides=[1, 1, 1, 1], padding='VALID') + conv_2_b
    
    #Activation.
    conv_2 = tf.nn.relu(conv_2)

    #Pooling. Input = 10x10x16. Output = 5x5x16.
    conv_2 = tf.nn.max_pool(conv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    #Flatten. Input = 5x5x16. Output = 400.
    flat = flatten(conv_2)
    
    #Layer 3: Fully Connected. Input = 400. Output = 120.
    fc_1_w = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
    fc_1_b = tf.Variable(tf.zeros(120))
    fc_1 = tf.matmul(flat, fc_1_w) + fc_1_b
    
    #Activation.
    fc_1 = tf.nn.relu(fc_1)
    
    #Layer 4: Fully Connected. Input = 120. Output = 84.
    fc_2_w = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc_2_b = tf.Variable(tf.zeros(84))
    fc_2 = tf.matmul(fc_1, fc_2_w) + fc_2_b
    
    #Activation.
    fc_2 = tf.nn.relu(fc_2)
    
    #Layer 5: Fully Connected. Input = 84. Output = 10.
    fc_log_w = tf.Variable(tf.truncated_normal(shape=(84, 43), mean=mu, stddev=sigma))
    fc_log_b = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc_2, fc_log_w) + fc_log_b
    
    return logits

##########################################################################################################################################

##########################################################################################################################################

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)

##########################################################################################################################################

##########################################################################################################################################

rate = 0.0009

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

##########################################################################################################################################

##########################################################################################################################################

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


##########################################################################################################################################

##########################################################################################################################################

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_valid, y_valid)
        training_accuracy = evaluate(X_train, y_train)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print("Training Accuracy = {:.3f}".format(training_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")


##########################################################################################################################################

##########################################################################################################################################

# Evaluate the model on the test set

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))


##########################################################################################################################################

##########################################################################################################################################

#Load the images and plot them.
import glob
import os
import matplotlib.image as mpimg

image_list = []
down_img = []
y_test_new = []

for filename in glob.glob('./web_signs/*.jpg'): 
    #Get sign image
    im = mpimg.imread(filename)
    #Get file name/ sign identifier
    sign_name = os.path.basename(filename)
    sign_name = int(sign_name.split("_")[0])
    #Print images from the web
    plt.figure()
    plt.xlabel(sign_name)
    plt.imshow(im)
    #Add sign identifier to y_test_new 
    y_test_new.append(sign_name)
    #Add sign image to image_list
    image_list.append(im)
    
for i in range(len(image_list)):
    #Resize sign images
    resized_image = cv.resize(image_list[i], (32, 32)) 
    down_img.append(resized_image)
    #Display resized images
    plt.figure()
    plt.xlabel(y_test_new[i])
    plt.imshow(down_img[i])


##########################################################################################################################################

##########################################################################################################################################

#Pre-Process new images.
temp_train = []

for i in range(len(down_img)):
    #Pre-process images
    temp_train.append(norm(down_img[i]))
    
x_test_new = np.array(temp_train)
print("temp_train normalization done")

##########################################################################################################################################

##########################################################################################################################################


### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    web_accuracy = evaluate(x_test_new, y_test_new)
    print("Web Accuracy = {:.3f}".format(web_accuracy))
    print("Test Accuracy = {:.3f}".format(test_accuracy))


##########################################################################################################################################

##########################################################################################################################################

#Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 

softmax = tf.nn.softmax(logits)

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    sMaxProb = sess.run(softmax, feed_dict={x: x_test_new})
    sMaxTop5 = sess.run(tf.nn.top_k(tf.constant(sMaxProb), k=5))

print(sMaxTop5)