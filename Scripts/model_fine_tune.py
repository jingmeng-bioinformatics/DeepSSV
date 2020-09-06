#!/usr/bin/env python

import argparse
import tensorflow as tf
import time

parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint_file', required=True, metavar='.data file', help='the full path to the checkpoint (.data) file for the pre-trained model')
parser.add_argument('--Mapping_information_file_fine_tune', required=True, metavar='file', help='file for fine-tuning with mapping information for validated somatic small variant sites')
parser.add_argument('--Mapping_information_file_validate', required=True, metavar='file', help='file for validation with mapping information for validated somatic small variant sites')
parser.add_argument('--batch_size', default=128, type=int, metavar='integer', help='total number of training examples in a single batch')
parser.add_argument('--buffer_size', default=600, type=int, metavar='integer', help='the maximum number of elements that will be buffered when prefetching')
parser.add_argument('--num_epochs', default=100, type=int, metavar='integer', help='the number of times the model sees the entire dataset')
parser.add_argument('--num_neurons_fc1', default=256, metavar='integer', help='the number of neurons in the first fully connected layer')
parser.add_argument('--learning_rate', default=0.0001, metavar=' ', help='learning rate for fine-tuning models')
parser.add_argument('--saved_model_path', required=True, metavar='path', help='the directory to save fine-tuned models')
parser.add_argument('--number_of_columns', required=True, type=int, help='the number of flanking genomic sites to the left or right of the candidate somatic site')

args = parser.parse_args()

def parse_example(line_batch, number_of_columns = args.number_of_columns):
    # define all values in each column to be floats
    record_defaults = [[1.0] for col in range(0, 2*args.number_of_columns + 1)]
    # specify the last column to be integer for the label
    record_defaults.append([1])
    # use TensorFlow csv decoder to covert the string tensor into a vector representation of features 
    content = tf.decode_csv(line_batch, record_defaults = record_defaults, field_delim = '\t')
    
    # pack all features into a tensor
    features = tf.stack(content[0:2*args.number_of_columns + 1])
    # transpose the feature tensor
    features = tf.transpose(features)
    label = content[-1][-1]
    
    # convert the label into a one-hot tensor
    label = tf.one_hot(indices = tf.cast(label, tf.int32), depth = 2)
    return features, label

def dataset_input_fn(file, batch_size = args.batch_size, buffer_size = args.buffer_size):
    dataset = tf.data.TextLineDataset(file)
    # combine 2805 lines into a single example
    dataset = dataset.batch(2805)
    dataset = dataset.map(parse_example)
    dataset = dataset.shuffle(buffer_size)
    
    # batch multiple examples
    dataset = dataset.batch(batch_size)
    # prefetch one batch
    dataset = dataset.prefetch(1)
    return dataset

def weight_variable(shape, name):
    # weight initialization, initializing weights with a small amount of noise for symmetry breaking
    # slightly positive initial bias to avoid "dead neurons" 
    initial = tf.truncated_normal(stddev=0.1, shape = shape, name = name) # stddev: standard deviation
    return tf.Variable(initial)

def bias_variable(shape, name):
    initial = tf.constant(0.0, shape = shape, name = name)
    return tf.Variable(initial)

def model_function(data_batch, label_batch, args):
    input_layer = tf.reshape(data_batch, [-1, 2*args.number_of_columns + 1, 2805])

    # the first convolutional layer
    W_conv1 = weight_variable([3, 2805, 32], name = 'W_conv1')
    b_conv1 = bias_variable([32], name = 'b_conv1')
    conv1_bn = tf.nn.conv1d(input_layer, W_conv1, stride = 1, padding='SAME') + b_conv1
    conv1 = tf.nn.relu(conv1_bn)
    
    # the second convolutional layer
    W_conv2 = weight_variable([3, 32, 32], name = 'W_conv2')
    b_conv2 = bias_variable([32], name = 'b_conv2')
    conv2_bn = tf.nn.conv1d(conv1, W_conv2, stride = 1, padding='SAME') + b_conv2 
    conv2 = tf.nn.relu(conv2_bn)

    # the pooling layer after the second convolutional layer
    pool2 = tf.layers.max_pooling1d(conv2, pool_size = 2, strides= 2, padding='same')

    # the third convolutional layer
    W_conv3 = weight_variable([3, 32, 64], name = 'W_conv3')
    b_conv3 = bias_variable([64], name = 'b_conv3')
    conv3_bn = tf.nn.conv1d(pool2, W_conv3, stride = 1, padding='SAME') + b_conv3
    conv3 = tf.nn.relu(conv3_bn)
    
    # the fourth convolutional layer
    W_conv4 = weight_variable([3, 64, 64], name = 'W_conv4')
    b_conv4 = bias_variable([64], name = 'b_conv4')
    conv4_bn = tf.nn.conv1d(conv3, W_conv4, stride = 1, padding='SAME') + b_conv4
    conv4 = tf.nn.relu(conv4_bn)

    # the pooling layer after the fourth convolutional layer
    pool4= tf.layers.max_pooling1d(conv4, pool_size = 2, strides= 2, padding='same')

    # densely connected (fully connected) layer           
    W_fc1 = weight_variable([64*(int(args.number_of_columns/2) + 1), args.num_neurons_fc1], name = 'W_fc1') 
    b_fc1 = bias_variable([args.num_neurons_fc1], name = 'b_fc1')
    pool4_flat = tf.reshape(pool4, [-1, 64*(int(args.number_of_columns/2) + 1)])
    fc1_bn = tf.matmul(pool4_flat, W_fc1) + b_fc1
    fc1 = tf.nn.relu(fc1_bn)

    # readout layer or softmax regression
    W_fc2 = weight_variable([args.num_neurons_fc1, 2], name = 'W_fc2')
    b_fc2 = bias_variable([2], name = 'b_fc2')
    y_conv = tf.matmul(fc1, W_fc2) + b_fc2

    # calculate loss
    loss = tf.losses.softmax_cross_entropy(onehot_labels = label_batch, logits = y_conv)
    # calculate accuracy
    equality = tf.equal(tf.argmax(y_conv, 1), tf.argmax(label_batch, 1))
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

    train_op = tf.train.AdamOptimizer(args.learning_rate).minimize(loss)
    return accuracy, loss, train_op
    
def main(args):
    # create the network manually as the restored model
    fine_tune_dataset = dataset_input_fn([args.Mapping_information_file_fine_tune])
    validate_dataset = dataset_input_fn([args.Mapping_information_file_validate])
    
    # create general iterator by the from_structure() method which needs the information of output data size/shape
    iterator = tf.data.Iterator.from_structure(fine_tune_dataset.output_types)
    data_batch, label_batch = iterator.get_next()

    # make datasets that we can initialize seperately.
    fine_tune_init_op = iterator.make_initializer(fine_tune_dataset)
    validate_init_op = iterator.make_initializer(validate_dataset)
    
    # call model_function
    accuracy, loss, train_op = model_function(data_batch, label_batch, args) 
    saver = tf.train.Saver(max_to_keep = 200)
  
    # with tf.Session(config = tf.ConfigProto(log_device_placement = True)) as sess:
    with tf.Session(config = tf.ConfigProto(gpu_options = tf.GPUOptions(visible_device_list = '0'))) as sess:
        # restore the values of the parameters in the pre-trained model
        saver.restore(sess, args.checkpoint_file)

        for i in range(args.num_epochs):
            start_time = time.time()
            k = 0
            acc_train = 0
            # initialize the iterator to fine_tune_dataset
            sess.run(fine_tune_init_op)
            while True:
                try:
                    accu, l, _ = sess.run([accuracy, loss, train_op])
                    k += 1
                    acc_train += accu
                    if k % 25 == 0:
                        print('Epoch: {}, step: {}, training loss: {:.3f}, training accuracy: {:.2f}%'.format(i, k, l, accu * 100))
                except tf.errors.OutOfRangeError:
                    break
            print('--- %s seconds ---' %(time.time() - start_time))
            
            # save the model after one epoch
            saver.save(sess, args.saved_model_path, global_step = i + 1)
            print('Epoch: {}, training loss: {:.3f}, training accuracy: {:.2f}%'.format(i, l, (acc_train / k) * 100))
            
            # set up the validation run
            acc_validate = 0
            j = 0
            start_time = time.time()

            # re-initialize the iterator, but this time with validate_dataset
            sess.run(validate_init_op)
            while True:
                try:
                    accu = sess.run(accuracy)
                    acc_validate += accu
                    j += 1
                except tf.errors.OutOfRangeError:
                    break
            print("validation accuracy: {:.2f}%".format((acc_validate / j) * 100))
            print('--- %s seconds ---' %(time.time() - start_time))

if __name__ == '__main__':
    main(args)
