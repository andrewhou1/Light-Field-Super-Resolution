import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import sys
import numpy as np
from PIL import Image as im
import random
import scipy.misc


x = tf.placeholder('float', [None, 17, 17, 1])
y = tf.placeholder('float', [None, 17, 17, 1])
'''weights={'W_conv1':tf.Variable(tf.truncated_normal([9, 9, 1, 64], mean = 0.0, stddev=0.0001)),
		 'W_conv2':tf.Variable(tf.truncated_normal([5, 5, 64, 32], mean = 0.0, stddev = 0.0001)),
		 'W_conv3':tf.Variable(tf.truncated_normal([5, 5, 32, 1], mean = 0.0, stddev = 0.0001))}
		
biases={'b_conv1':tf.Variable(tf.truncated_normal([64], mean = 0.0, stddev = 0.0001)),
		'b_conv2':tf.Variable(tf.truncated_normal([32], mean = 0.0, stddev = 0.0001)),
		'b_conv3':tf.Variable(tf.truncated_normal([1], mean = 0.0, stddev = 0.0001))}'''
		
def convolutional_NN(x):
	x = tf.reshape(x, shape=[-1, 17, 17, 1])
	'''conv1 = tf.nn.relu(conv2d(x, weights['W_conv1'])+biases['b_conv1'])
	conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2'])+biases['b_conv2'])
	conv3 = conv2d(conv2, weights['W_conv3']+biases['b_conv3'])'''
	conv1 = tf.nn.relu(conv2d(x, W1)+B1)
	conv2 = tf.nn.relu(conv2d(conv1, W2)+B2)
	conv3 = conv2d(conv2, W3)+B3
	#conv3_drop = tf.nn.dropout(conv3, keep_prob)
	#output = tf.add(x, conv3_drop)
	
	#return (output, conv3_drop, conv3, conv2, conv1)
	return (conv3, conv2, conv1)
	
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
	
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.import_meta_graph('model_withnewloss/my_model50.meta')
saver.restore(sess, tf.train.latest_checkpoint('model_withnewloss/.'))

graph = tf.get_default_graph()
W1 = graph.get_tensor_by_name('W1:0')
W2 = graph.get_tensor_by_name('W2:0')
W3 = graph.get_tensor_by_name('W3:0')
B1 = graph.get_tensor_by_name('b1:0')
B2 = graph.get_tensor_by_name('b2:0')
B3 = graph.get_tensor_by_name('b3:0')
#print(sess.run(W1))

(predicted_residual, conv2, conv1) = convolutional_NN(x)
loss = tf.reduce_mean(tf.squared_difference(y, predicted_residual))
	
filelist_inputs = sorted(os.listdir('upsampled_EPIs2'))
for img in filelist_inputs[:]:
	if not(img.endswith(".png")):
		filelist_inputs.remove(img)
			
filelist_outputs = sorted(os.listdir('Matlab_Training_Data\groundtruthblurred\groundtruthblurred'))
for img in filelist_outputs[:]:
	if not(img.endswith(".png")):
		filelist_outputs.remove(img)

tmp_filelist_outputs = []		
for i in range(17):
	for j in range(800):
		if(i % 2 == 0):
			img = filelist_outputs[i*800+j]
			tmp_filelist_outputs.append(img)
	
filelist_outputs = tmp_filelist_outputs

lowestYValue = 10000
highestYValue = -10000
for file in range(len(filelist_inputs)):
	input_filename = "upsampled_EPIs2/"+filelist_inputs[file]
	image = im.open(input_filename)
	ycbcr = image.convert('YCbCr')
	trueimg = np.ndarray((image.size[1], image.size[0], 3), 'u1', ycbcr.tobytes())
	ychannel = trueimg[:, :, 0]
	dims = np.shape(ychannel)
	input_subEPIs = np.zeros((100, 17, 17, 1))
	colsRemaining = dims[1]
	count = 0
	startIndex = 0
	stride = 14

	while(colsRemaining >= 17):
		EPI = ychannel[:, startIndex:startIndex+17]
		EPI = np.reshape(EPI, (17, 17, 1))
		input_subEPIs[count, :, :, :] = EPI
		count += 1
		startIndex += 14
		colsRemaining -= 14

	input_subEPIs[99, :, :, :] = np.reshape(ychannel[:, startIndex-3:startIndex+colsRemaining], (17, 17, 1))
			
	output_filename = "Matlab_Training_Data/groundtruthblurred/groundtruthblurred/"+filelist_outputs[file]
	image = im.open(output_filename)
	ycbcr = image.convert('YCbCr')
	trueimg = np.ndarray((image.size[1], image.size[0], 3), 'u1', ycbcr.tobytes())
	ychannel = trueimg[:, :, 0]
	dims = np.shape(ychannel)
	output_subEPIs = np.zeros((100, 17, 17, 1))
	colsRemaining = dims[1]
	count = 0
	startIndex=0
	stride = 14

	while(colsRemaining >= 17):
		EPI = ychannel[:, startIndex:startIndex+17]
		EPI = np.reshape(EPI, (17, 17, 1))
		output_subEPIs[count, :, :, :] = EPI
		count += 1
		startIndex += 14
		colsRemaining -= 14

	output_subEPIs[99, :, :, :] = np.reshape(ychannel[:, startIndex-3:startIndex+colsRemaining], (17, 17, 1))
	input_subEPIs = input_subEPIs/255
	output_subEPIs = output_subEPIs/255	

	#ls, out, residual_drop, residual, c2, c1 = sess.run([loss, predictions, conv3_drop, conv3, conv2, conv1], feed_dict = {x:input_subEPIs, y:output_subEPIs})
	ls, residual, c2, c1 = sess.run([loss, predicted_residual, conv2, conv1], feed_dict = {x:input_subEPIs, y:output_subEPIs})
	
	out = input_subEPIs+residual
	maxPixel = np.amax(out)
	minPixel = np.amin(out)
	if(maxPixel > highestYValue):
		highestYValue = maxPixel
		
	if(minPixel < lowestYValue):
		lowestYValue = minPixel
		
print(highestYValue)
print(lowestYValue)

first_row_residuals = np.zeros((17, 1400, 800))
for file in range(len(filelist_inputs)):
	input_filename = "upsampled_EPIs2/"+filelist_inputs[file]
	image = im.open(input_filename)
	ycbcr = image.convert('YCbCr')
	trueimg = np.ndarray((image.size[1], image.size[0], 3), 'u1', ycbcr.tobytes())
	ychannel = trueimg[:, :, 0]
	dims = np.shape(ychannel)
	input_subEPIs = np.zeros((100, 17, 17, 1))
	colsRemaining = dims[1]
	count = 0
	startIndex = 0
	stride = 14

	while(colsRemaining >= 17):
		EPI = ychannel[:, startIndex:startIndex+17]
		EPI = np.reshape(EPI, (17, 17, 1))
		input_subEPIs[count, :, :, :] = EPI
		count += 1
		startIndex += 14
		colsRemaining -= 14

	input_subEPIs[99, :, :, :] = np.reshape(ychannel[:, startIndex-3:startIndex+colsRemaining], (17, 17, 1))
			
	output_filename = "Matlab_Training_Data/groundtruthblurred/groundtruthblurred/"+filelist_outputs[file]
	image = im.open(output_filename)
	ycbcr = image.convert('YCbCr')
	trueimg = np.ndarray((image.size[1], image.size[0], 3), 'u1', ycbcr.tobytes())
	ychannel = trueimg[:, :, 0]
	dims = np.shape(ychannel)
	output_subEPIs = np.zeros((100, 17, 17, 1))
	colsRemaining = dims[1]
	count = 0
	startIndex=0
	stride = 14

	while(colsRemaining >= 17):
		EPI = ychannel[:, startIndex:startIndex+17]
		EPI = np.reshape(EPI, (17, 17, 1))
		output_subEPIs[count, :, :, :] = EPI
		count += 1
		startIndex += 14
		colsRemaining -= 14

	output_subEPIs[99, :, :, :] = np.reshape(ychannel[:, startIndex-3:startIndex+colsRemaining], (17, 17, 1))
	input_subEPIs = input_subEPIs/255
	output_subEPIs = output_subEPIs/255	
	
	ls, residual, c2, c1 = sess.run([loss, predicted_residual, conv2, conv1], feed_dict = {x:input_subEPIs, y:output_subEPIs})
	output_img = np.zeros(dims)
	residual_img = np.zeros(dims)
	sub_EPI_dims = np.shape(residual)
	begin = 0
	end = 17
	
	out = input_subEPIs+residual
	
	for i in range(sub_EPI_dims[0]):
		if(i == 0):
			patch = np.reshape(out[i, :, :, :], (17, 17))
			dims = np.shape(patch)
			residual_patch = np.reshape(residual[i, :, :, :], (17,17))
			output_img[:, begin:end] = patch
			residual_img[:, begin:end] = residual_patch
			begin += 17
			end += 14
		elif(i == 99):
			last_col = dims[1]-1
			remainingCols = last_col-begin+1
			patch = np.reshape(out[i, :, 3:remainingCols+3, :], (17, remainingCols))
			residual_patch = np.reshape(residual[i, :, 3:remainingCols+3, :], (17, remainingCols))
			output_img[:, begin:begin+remainingCols] = patch
			residual_img[:, begin:begin+remainingCols] = residual_patch	
		else:
			patch = np.reshape(out[i, :, 3:17, :], (17, 14))
			dims = np.shape(patch)
			residual_patch = np.reshape(residual[i, :, 3:17, :], (17,14))
			output_img[:, begin:end] = patch
			residual_img[:, begin:end] = residual_patch
			begin += 14
			end += 14

	scipy.misc.toimage(output_img, cmax=1.0, cmin=0.0).save('output_newloss/'+'output_'+filelist_outputs[file])
	scipy.misc.toimage(residual_img).save('residuals_newloss/'+'residual_'+filelist_outputs[file])