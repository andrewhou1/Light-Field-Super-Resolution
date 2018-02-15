import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import sys
import numpy as np
from PIL import Image as im
import random
import scipy.misc
from skimage import io, color


batch_size = 64
#keep_prob = 0.8
#g = tf.Graph()
#with g.as_default():
x = tf.placeholder('float', [None, 17, 17, 1])
y = tf.placeholder('float', [None, 17, 17, 1])
weights={'W_conv1':tf.Variable(tf.truncated_normal([9, 9, 1, 64], mean = 0.0, stddev=0.0001), name='W1'),
		 'W_conv2':tf.Variable(tf.truncated_normal([5, 5, 64, 32], mean = 0.0, stddev = 0.0001), name='W2'),
		 'W_conv3':tf.Variable(tf.truncated_normal([5, 5, 32, 1], mean = 0.0, stddev = 0.0001), name='W3')}
		
biases={'b_conv1':tf.Variable(tf.truncated_normal([64], mean = 0.0, stddev = 0.0001), name='b1'),
		'b_conv2':tf.Variable(tf.truncated_normal([32], mean = 0.0, stddev = 0.0001), name='b2'),
		'b_conv3':tf.Variable(tf.truncated_normal([1], mean = 0.0, stddev = 0.0001), name='b3')}

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def relu(x, alpha=0.01, max_value=None):
    return tf.maximum(alpha*x, x)
	
def convolutional_NN(x):
	x = tf.reshape(x, shape=[-1, 17, 17, 1])
	conv1 = tf.nn.relu(conv2d(x, weights['W_conv1'])+biases['b_conv1'])
	conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2'])+biases['b_conv2'])
	conv3 = conv2d(conv2, weights['W_conv3'])+biases['b_conv3']
	#conv3_drop = tf.nn.dropout(conv3, keep_prob)
	#output = tf.add(x, conv3_drop)
	#output = tf.add(x, conv3)
	
	#return (output, conv3_drop, conv3, conv2, conv1)
	return (conv3, conv2, conv1)

def generate_inputs(inputs_dir, outputs_dir, inputs_dir2, outputs_dir2):
	filelist_inputs = sorted(os.listdir(inputs_dir))
	for img in filelist_inputs[:]:
		if not(img.endswith(".png")):
			filelist_inputs.remove(img)

	input_sub_EPIs = np.zeros((100*len(filelist_inputs), 17, 17, 1))
	
	for i in range(len(filelist_inputs)):
		filename = inputs_dir+"/"+filelist_inputs[i]
		image = io.imread(filename)
		lab = color.rgb2lab(image)
		Lchannel =  lab[:, :, 0]
		dims = np.shape(Lchannel)
		subEPIs = np.zeros((100, 17, 17, 1))
		colsRemaining = dims[1]
		count = 0
		startIndex = 0
		stride = 14

		while(colsRemaining >= 17):
			EPI = Lchannel[:, startIndex:startIndex+17]
			EPI = np.reshape(EPI, (17, 17, 1))
			subEPIs[count, :, :, :] = EPI
			count += 1
			startIndex += 14
			colsRemaining -= 14
		#Get rid of last subEPI and replace it with last 17 columns of image
		subEPIs[99, :, :, :] = np.reshape(Lchannel[:, startIndex-3:startIndex+colsRemaining], (17, 17, 1))
		
		input_sub_EPIs[100*i:100*(i+1), :, :, :] = subEPIs
		
	filelist_inputs2 = sorted(os.listdir(inputs_dir2))
	for img in filelist_inputs2[:]:
		if not(img.endswith(".png")):
			filelist_inputs2.remove(img)
			
	input_sub_EPIs2 = np.zeros((100*len(filelist_inputs2), 17, 17, 1))
	
	for i in range(len(filelist_inputs2)):
		filename = inputs_dir2+"/"+filelist_inputs2[i]
		image = io.imread(filename)
		lab = color.rgb2lab(image)
		Lchannel =  lab[:, :, 0]
		dims = np.shape(Lchannel)
		subEPIs = np.zeros((100, 17, 17, 1))
		colsRemaining = dims[1]
		count = 0
		startIndex = 0
		stride = 14

		while(colsRemaining >= 17):
			EPI = Lchannel[:, startIndex:startIndex+17]
			EPI = np.reshape(EPI, (17, 17, 1))
			subEPIs[count, :, :, :] = EPI
			count += 1
			startIndex += 14
			colsRemaining -= 14
		subEPIs[99, :, :, :] = np.reshape(Lchannel[:, startIndex-3:startIndex+colsRemaining], (17, 17, 1))
		
		input_sub_EPIs2[100*i:100*(i+1), :, :, :] = subEPIs
		
	#all_inputs = np.concatenate((input_sub_EPIs, input_sub_EPIs2), axis=0)
	
	filelist_outputs = sorted(os.listdir(outputs_dir))
	for img in filelist_outputs[:]:
		if not(img.endswith(".png")):
			filelist_outputs.remove(img)
	
	output_sub_EPIs = np.zeros((100*len(filelist_outputs), 17, 17, 1))
	
	for i in range(len(filelist_outputs)):
		filename = outputs_dir+"/"+filelist_outputs[i]
		image = io.imread(filename)
		lab = color.rgb2lab(image)
		Lchannel =  lab[:, :, 0]
		dims = np.shape(Lchannel)
		subEPIs = np.zeros((100, 17, 17, 1))
		colsRemaining = dims[1]
		count = 0
		startIndex = 0
		stride = 14

		while(colsRemaining >= 17):
			EPI = Lchannel[:, startIndex:startIndex+17]
			EPI = np.reshape(EPI, (17, 17, 1))
			subEPIs[count, :, :, :] = EPI
			count += 1
			startIndex += 14
			colsRemaining -= 14
		subEPIs[99, :, :, :] = np.reshape(Lchannel[:, startIndex-3:startIndex+colsRemaining], (17, 17, 1))
		output_sub_EPIs[100*i:100*(i+1), :, :, :] = subEPIs
		
	
	filelist_outputs2 = sorted(os.listdir(outputs_dir2))
	for img in filelist_outputs2[:]:
		if not(img.endswith(".png")):
			filelist_outputs2.remove(img)
	
	output_sub_EPIs2 = np.zeros((100*len(filelist_outputs2), 17, 17, 1))
	
	for i in range(len(filelist_outputs2)):
		filename = outputs_dir+"/"+filelist_outputs2[i]
		image = io.imread(filename)
		lab = color.rgb2lab(image)
		Lchannel =  lab[:, :, 0]
		dims = np.shape(Lchannel)
		subEPIs = np.zeros((100, 17, 17, 1))
		colsRemaining = dims[1]
		count = 0
		startIndex = 0
		stride = 14

		while(colsRemaining >= 17):
			EPI = Lchannel[:, startIndex:startIndex+17]
			EPI = np.reshape(EPI, (17, 17, 1))
			subEPIs[count, :, :, :] = EPI
			count += 1
			startIndex += 14
			colsRemaining -= 14
		subEPIs[99, :, :, :] = np.reshape(Lchannel[:, startIndex-3:startIndex+colsRemaining], (17, 17, 1))
		output_sub_EPIs2[100*i:100*(i+1), :, :, :] = subEPIs
	#all_outputs = np.concatenate((output_sub_EPIs, output_sub_EPIs2), axis=0)
	print(len(output_sub_EPIs)/2)
	indices_to_get_from_2 = random.sample(range(0, len(output_sub_EPIs2)), int(len(output_sub_EPIs2)/2))
	print("Here")
	input_sub_EPIs[indices_to_get_from_2] = input_sub_EPIs[indices_to_get_from_2]
	output_sub_EPIs[indices_to_get_from_2] = output_sub_EPIs2[indices_to_get_from_2]

	return (input_sub_EPIs, output_sub_EPIs)
	
def trainingDataVariation(inputs, outputs):
	dims_input = np.shape(inputs)
	dims_output = np.shape(outputs)
	for i in range(dims_input[0]):
		flip = random.random()
		if(flip > 0.5):
			input_img = inputs[i, :, :, :]
			input_img = np.reshape(input_img, (17, 17))
			input_img = np.fliplr(input_img)
			input_img = np.reshape(input_img, (17, 17, 1))
			inputs[i, :, :, :] = input_img
			output_img = outputs[i, :, :, :]
			output_img = np.reshape(output_img, (17, 17))
			output_img = np.fliplr(output_img)
			output_img = np.reshape(output_img, (17, 17, 1))
			outputs[i, :, :, :] = output_img
		input_img = inputs[i, :, :, :]
		'''mean = 0
		sigma = 0.01
		gaussian_noise = np.random.normal(mean, sigma, (17,17,1))
		input_img = input_img+gaussian_noise
		inputs[i, :, :, :] = input_img'''
				
	return (inputs, outputs)
	
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
	

def main():
	(in_sub_EPIs, out_sub_EPIs) = generate_inputs(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
	print(len(in_sub_EPIs))
	print(len(out_sub_EPIs))
	#in_sub_EPIs = in_sub_EPIs/255
	#out_sub_EPIs = out_sub_EPIs/255
	(input_sub_EPIs, output_sub_EPIs) = trainingDataVariation(in_sub_EPIs, out_sub_EPIs)
	input_sub_EPIs = (input_sub_EPIs-50)/50
	output_sub_EPIs = (output_sub_EPIs-50)/50
	(input_sub_EPIs, output_sub_EPIs) = unison_shuffled_copies(input_sub_EPIs, output_sub_EPIs)
	#print(input_sub_EPIs)
	#print(output_sub_EPIs)
	#with g.as_default():
	#(predictions, conv3_drop, conv3, conv2, conv1) = convolutional_NN(x)
	(predicted_residual, conv2, conv1) = convolutional_NN(x)
	loss = tf.reduce_mean(tf.squared_difference(y, predicted_residual))
	learning_rate = 0.0001
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
	#gvs = optimizer.compute_gradients(loss)
	#capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
	#train_op = optimizer.apply_gradients(capped_gvs)
	num_epochs = 50
	dims = np.shape(in_sub_EPIs)
	#dims = np.shape(input_sub_EPIs)
	num_EPIs = dims[0]
	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
	
		for epoch in range(num_epochs):
			epoch_loss = 0
			(input_sub_EPIs, output_sub_EPIs) = trainingDataVariation(input_sub_EPIs, output_sub_EPIs)
			(input_sub_EPIs, output_sub_EPIs) = unison_shuffled_copies(input_sub_EPIs, output_sub_EPIs)
			for i in range(int(num_EPIs/batch_size)):
				epoch_x = input_sub_EPIs[batch_size*i:batch_size*(i+1), :, :, :]
				epoch_y = output_sub_EPIs[batch_size*i:batch_size*(i+1), :, :, :]-epoch_x
				#_, ls, c3_d = sess.run([optimizer, loss, conv3_drop], feed_dict = {x:epoch_x, y:epoch_y})
				_, ls, c3 = sess.run([optimizer, loss, predicted_residual], feed_dict = {x:epoch_x, y:epoch_y})
				#print(gradients)
				#print('\n')
				#print(i)
				#print(c1)
				#print(np.sum(c3_d))
				#print('\n')
				#print('\n')
				#print('\n')
				epoch_loss += ls

			print('Epoch ', epoch, 'completed. Loss: ', epoch_loss)
			#weights['W_conv1'] = tf.Print(weights['W_conv1'], [weights['W_conv1']])
			#weights['W_conv2'] = tf.Print(weights['W_conv2'], [weights['W_conv2']])
			#weights['W_conv3'] = tf.Print(weights['W_conv3'], [weights['W_conv3']])
			#biases['b_conv1'] = tf.Print(biases['b_conv1'], [biases['b_conv1']])
			#biases['b_conv2'] = tf.Print(biases['b_conv2'], [biases['b_conv2']])
			#biases['b_conv3'] = tf.Print(biases['b_conv3'], [biases['b_conv3']])
		saver = tf.train.Saver(tf.global_variables())
		saver.save(sess, "model_Lab_space/my_model50")
			
if __name__ == "__main__":
	main()	



