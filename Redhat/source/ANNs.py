import os
import sys
import tensorflow as tf
#from PIL import Image
#from util.freeze_graph import freeze_graph
import numpy as np
import DataLoader

BATCH_SIZE = 50000
NUM_EPOCHS = 500
SEED = None

class ANNsModel:

    attributes_count = 0
    stddev_value = 0.1
    init_learn_rate = 0.8
    decay_rate = 0.9999

    model_save_dir = ''
    model_save_file_name = 'train_result'

    label_list = None

    train_size = 0
    validation_size = 0
    test_size = 0
    labelCount = 0

    train_data = None
    train_labels = None

    validation_data = None
    validation_labels = None

    test_data = None
    test_labels = None

    # Model parameters

    fc1_weights = None
    fc1_biases = None

    fc2_weights = None
    fc2_biases = None

    fc3_weights = None
    fc3_biases = None

    fc4_weights = None
    fc4_biases = None

    def InitVars(self):
        self.label_list = [0,1]
        self.labelCount = len(self.label_list)

        # Set parameters
        self.fc1_weights = tf.Variable(  # fully connected
            tf.truncated_normal(
                [self.attributes_count, self.attributes_count],
                stddev=self.stddev_value,
                seed=SEED), name='fc1_weights')
        self.fc1_biases = tf.Variable(tf.constant(0.1, shape=[self.attributes_count]), name='fc1_biases')
        '''
        self.fc2_weights = tf.Variable(  # fully connected
            tf.truncated_normal(
                [self.attributes_count, self.attributes_count],
                mean=1,
                stddev=self.stddev_value,
                seed=SEED), name='fc2_weights')
        self.fc2_biases = tf.Variable(tf.constant(0.1, shape=[self.attributes_count]), name='fc2_biases')

        self.fc3_weights = tf.Variable(
            tf.truncated_normal([self.attributes_count, self.attributes_count],
                                stddev=self.stddev_value,
                                seed=SEED), name='fc3_weights')
        self.fc3_biases = tf.Variable(tf.constant(self.stddev_value, shape=[self.attributes_count]), name='fc3_biases')
        '''
        self.fc4_weights = tf.Variable(
            tf.truncated_normal([self.attributes_count, self.labelCount],
                                mean=1,
                                stddev=self.stddev_value,
                                seed=SEED), name='fc4_weights')
        self.fc4_biases = tf.Variable(tf.constant(0.1, shape=[self.labelCount]), name='fc4_biases')

    def LoadData(self):
        self.train_data, activity_tem, self.train_labels = DataLoader.LoadTraingData(True)
        self.validation_data, activity_tem, self.validation_labels = DataLoader.LoadTrainValidationData(True)
        self.test_data, activity_tem, self.test_labels = DataLoader.LoadTrainTestingData(True)
        #print "self.train_labels before: ", self.train_labels[:10]
        self.train_labels = (np.arange(2) == self.train_labels[:, None]).astype(np.float32)
        #print "self.train_labels ",self.train_labels[:10,:]
        self.validation_labels = (np.arange(2) == self.validation_labels[:, None]).astype(np.float32)
        self.test_labels = (np.arange(2) == self.test_labels[:, None]).astype(np.float32)

        self.validation_size = self.validation_data.shape[0]
        self.test_size = self.test_data.shape[0]
        self.train_size = self.train_data.shape[0]

        self.labelCount = 2
        self.attributes_count =  self.train_data.shape[1]
        print self.attributes_count

    def CreateModel(self, data, train=False):
        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        hidden1 = tf.nn.relu(tf.matmul(data, self.fc1_weights) + self.fc1_biases)
        # Add a 50% dropout during training only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.

        if train:
            hidden1 = tf.nn.dropout(hidden1, 0.999, seed=SEED)
        '''
        hidden2 = tf.nn.relu(tf.matmul(hidden1, self.fc2_weights) + self.fc2_biases)
        if train:
            hidden2 = tf.nn.dropout(hidden2, 0.999, seed=SEED)

        hidden3 = tf.nn.sigmoid(tf.matmul(hidden2, self.fc3_weights) + self.fc3_biases)
        if train:
            hidden3 = tf.nn.dropout(hidden3, 0.5, seed=SEED)

        return tf.matmul(hidden3, self.fc4_weights) + self.fc4_biases
        '''
        return tf.matmul(hidden1, self.fc4_weights) + self.fc4_biases
    def TrainModel(self):
        train_data_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, self.attributes_count))
        train_labels_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, self.labelCount))
        validation_data_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE,self.attributes_count))
        validation_labels_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, self.labelCount))

        logits = self.CreateModel(train_data_node, True)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, train_labels_node))

        '''
        # L2 regularization for the fully connected parameters.
        regularizers = (tf.nn.l2_loss(self.fc1_weights) + tf.nn.l2_loss(self.fc1_biases) +
                        tf.nn.l2_loss(self.fc2_weights) + tf.nn.l2_loss(self.fc2_biases)+
                        tf.nn.l2_loss(self.fc3_weights) + tf.nn.l2_loss(self.fc3_biases) +
                        tf.nn.l2_loss(self.fc4_weights) + tf.nn.l2_loss(self.fc4_biases))
        # Add the regularization term to the loss.
        loss += 5e-3 * regularizers
        '''
        # Optimizer: set up a variable that's incremented once per batch and
        # controls the learning rate decay.
        batch = tf.Variable(0)
        # Decay once per epoch, using an exponential schedule starting at 0.01.
        learning_rate = tf.train.exponential_decay(
            self.init_learn_rate,  # Base learning rate.
            batch * BATCH_SIZE,  # Current index into the dataset.
            self.train_size,  # Decay step.
            self.decay_rate,  # Decay rate.
            staircase=True)
        # Use simple momentum for the optimization.
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=batch)
        # Predictions for the minibatch, validation set and test set.
        train_prediction = tf.nn.softmax(logits)
        validation_prediction = tf.nn.softmax(self.CreateModel(validation_data_node))

        # vars to be saved
        store_list = [self.fc1_weights, self.fc1_biases,

                      #self.fc2_weights,self.fc2_biases,
                      #self.fc3_weights,self.fc3_biases,
                      self.fc4_weights, self.fc4_biases]

        # Create saver
        saver = tf.train.Saver(store_list);

        def CaculateErrorRate(session, dataList, labels):
            data_size = dataList.shape[0]
            errorCount = 0;
            for step in xrange(int(data_size / BATCH_SIZE)):
                offset = (step * BATCH_SIZE)
                batch_data = dataList[offset:(offset + BATCH_SIZE), :]
                batch_labels = labels[offset:(offset + BATCH_SIZE)]
                feed_dict = {validation_data_node: batch_data,
                             validation_labels_node: batch_labels}
                # Run the graph and fetch some of the nodes.
                # print batch_data.shape
                # print batch_labels.shape
                # print train_labels
                validation_prediction_result = session.run(validation_prediction, feed_dict=feed_dict)
                errorCount += np.sum(np.argmax(validation_prediction_result, 1) != np.argmax(batch_labels, 1))
                '''
                if step ==0:
                    print "Prediction: " ,np.argmax(validation_prediction_result, 1)[:20]
                    print "Real labels: " , labels[:20,:]

                '''
            return errorCount * 100.0 / data_size

        with tf.Session() as s:

            tf.initialize_all_variables().run()

            ckpt = tf.train.get_checkpoint_state(self.model_save_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print "find the checkpoing file"
                saver.restore(s, ckpt.model_checkpoint_path)

            print 'Initialized!'
            # Loop through training steps.
            for step in xrange(int(NUM_EPOCHS * self.train_size / BATCH_SIZE)):
                # Compute the offset of the current minibatch in the data.
                # Note that we could use better randomization across epochs.
                offset = (step * BATCH_SIZE) % (self.train_size - BATCH_SIZE)
                batch_data = self.train_data[offset:(offset + BATCH_SIZE), :]
                batch_labels = self.train_labels[offset:(offset + BATCH_SIZE)]
                # This dictionary maps the batch data (as a numpy array) to the
                # node in the graph is should be fed to.
                # print batch_data.shape
                feed_dict = {train_data_node: batch_data,
                             train_labels_node: batch_labels}
                # Run the graph and fetch some of the nodes.
                print batch_data.shape
                print batch_labels.shape
                _, l, lr, predictions = s.run(
                    [optimizer, loss, learning_rate, train_prediction],
                    feed_dict=feed_dict)

                if step % 1 == 0:

                    print 'Epoch %.2f' % (float(step) * BATCH_SIZE / self.train_size)
                    print 'Minibatch loss: %.3f, learning rate: %.6f' % (l, lr)

                if step % 100 == 0 and step != 0:
                    saver.save(s, save_path=os.path.join(self.model_save_dir, self.model_save_file_name))
                    print 'Validation error: %.1f%%' % CaculateErrorRate(s, self.validation_data,
                                                                         self.validation_labels)

                sys.stdout.flush()

            saver.save(s, save_path=os.path.join(self.model_save_dir, self.model_save_file_name))
            # saver.save(s,save_path='../models/producttype/train_result')
            # Finally print the result!
            test_error = CaculateErrorRate(s, self.test_data, self.test_labels)
            print 'Test error: %.1f%%' % test_error

    def RestoreParameters(self, session):
        # vars to be saved
        store_list = [self.fc1_weights, self.fc1_biases,
                      #self.fc2_weights,self.fc2_biases,
                      #self.fc3_weights, self.fc3_biases,
                      self.fc4_weights, self.fc4_biases]
        restorer = tf.train.Saver(store_list)
        restorer.restore(session, save_path=os.path.join(self.model_save_dir, self.model_save_file_name))

    def Predict(self, data, session=None):
        check_data_node = tf.placeholder(tf.float32, shape=(1,self.attributes_count))
        prediction = tf.nn.softmax(self.CreateModel(check_data_node))
        s = session
        if s == None:
            s = tf.Session()
            self.RestoreParameters(s)
        # with tf.Session() as s:
        # self.RestoreParameters(s)
        reshaped_data = np.reshape(data, (1, self.attributes_count))
        feed_dict = {check_data_node: reshaped_data}
        prediction_result = s.run(prediction, feed_dict=feed_dict)
        result_index = np.argmax(prediction_result, 1)[0]
        print "Prediction:", prediction_result
        print "result_index:", result_index
        if session == None:
            s.close()
        return self.label_list[result_index]


def Train():
    model = ANNsModel()
    model.LoadData()
    model.InitVars()
    model.model_save_dir = '../predict_result'
    model.model_save_file_name='ANNModel'
    model.TrainModel()

def Predict():
    model = ANNsModel()
    model.LoadData()
    model.InitVars()
    model.model_save_dir = '../predict_result'
    model.model_save_file_name = 'ANNModel'
    s = tf.Session()
    model.RestoreParameters(s)
    X_test, activities_test,y = DataLoader.LoadTrainValidationData()
    for i in range(100):
        model.Predict(X_test[i],s)
        print "actual %s"%y[i]
    s.close()
Train()