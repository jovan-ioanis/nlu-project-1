import numpy as np
import tensorflow as tf
import os
import sys, getopt
import preprocessing
import time
import math
import load_embeddings 
from graphs import build_training_graph, VOCABULARY_SIZE, CELL_SIZE, BATCH_SIZE, MAX_SENTENCE_LENGTH, LEARNING_RATE, WORD_EMBEDDING_SIZE, SEQUENCE_LENGTH

TRAINING_DATA_PATH = 'data/sentences.train'
LOG_DIRECTORY = "logs/"
NUM_EPOCHS = 10
CHECKPOINT_FREQUENCY = 15000 # After how many batches we will save a checkpoint
VALIDATION_SUMMARY_FREQUENCY = 200 # How often we will do a validation step on unseen data
TRAIN_SUMMARY_FREQUENCY = 100 # How often we will output a summary with the training_step

def shuffle_iterator(raw_data, batch_size, sequence_length=SEQUENCE_LENGTH):
    """
        Loops through the raw_data, returning it in batches.
        Additionally shuffles the raw_data.
        
        raw_data       A list of raw sentences (of word indices)
        batch_size     Size of the batches to return
        sequence_length Length of the sentences in raw_data
    """
    
    np.random.shuffle(raw_data)
    return data_iterator(raw_data, batch_size, sequence_length=SEQUENCE_LENGTH)

def data_iterator(raw_data, batch_size, sequence_length=SEQUENCE_LENGTH):
    """
        Loops through the raw_data, returning it in batches

        raw_data       A list of raw sentences (of word indices)
        batch_size     Size of the batches to return
        sequence_length Length of the sentences in raw_data
    """

    data_len = len(raw_data)
    num_batches = data_len // batch_size
    r = data_len % batch_size

    if r: # If the data size is not a multiple of the batch size, we randomly remove some sentences until they align
        mask = np.ones(data_len, dtype=bool)
        mask[np.random.choice(data_len, r, replace=False)] = False
        raw_data = raw_data[mask,:]


    for i in range(num_batches):
        if i % VALIDATION_SUMMARY_FREQUENCY == 0 : # Regularly print the current status
            print("Fetching batch {} of out {}".format(i, num_batches))

        data = raw_data[(batch_size*i):(batch_size*(i+1)),:]
        x = data[:, :sequence_length]
        y = data[:, 1:]
        yield (x,y)

def train_network(operators, sentences_array, num_epochs, vocabulary,
                    configProto=None,
                    num_steps = MAX_SENTENCE_LENGTH,
                    batch_size = BATCH_SIZE,
                    state_size = CELL_SIZE,
                    checkpoint_filename=None,
                    useWord2Vec=False):

    """
        Trains the network using a given graph

        operators           Dictionary of graph operators to execute
        sentences_array     Input sentences (in index form)
        num_epochs          Number of epochs to train for
        vocabulary          Word2Index dictionary of our vocabulary
        configProto         Session configuration for tensorflow
        num_steps           Number of steps of our RNN / Sequence length
        batch_size          Number of sentences per batch
        state_size          Size of the hidden state in an RNN cell
        checkpoint_filename Name of the file to save the graph.
                            Set to None if no saving is required.
        useWord2Vec         Indicates whether to load word embeddings
                            from the provided word2vec file.
    """

    with tf.Session(config=configProto) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)
        global_step = 1

        # Init Tensorboard summaries. This will save Tensorboard information into a different folder at each run.
        timestamp = str(math.trunc(time.time()))
        train_writer = tf.summary.FileWriter("{}{}-{}-training".format(LOG_DIRECTORY, timestamp, checkpoint_filename), graph=tf.get_default_graph())
        validation_writer = tf.summary.FileWriter("{}{}-{}-validation".format(LOG_DIRECTORY, timestamp, checkpoint_filename), graph=tf.get_default_graph())

        '''
            If the flag is activated, load word2vec embeddings. It is placed just before training
            since it requires session object. Also, since it needs W_embed tensor, we reach it from "word_embedding" variable_scope.

            load_embeddings.py requires installing gensim package!
        '''
        if useWord2Vec == True :
            with tf.variable_scope("word_embedding"):
                tf.get_variable_scope().reuse_variables()
                W_embed = tf.get_variable("W_embed")
            load_embeddings.load_embedding( session=sess,
                                            vocab=vocabulary,
                                            emb=W_embed,
                                            path="wordembeddings-dim100.word2vec",
                                            dim_embedding=100,
                                            vocab_size=VOCABULARY_SIZE)


        # Setup the feed dictionary and fetch the operators to execute
        zero_state = np.zeros([batch_size, state_size]) 
        feed_dict={operators['init_state_a']:zero_state, operators['init_state_b']:zero_state} # For training we always feed zero states at the beginning of each sentence
        summary_op = operators['summary_op']
        train_step = operators['train_step']
        x = operators['x']
        y = operators['y']

        for epoch in range(num_epochs):
            print("Starting epoch {}".format(epoch))
            for X, Y in shuffle_iterator(sentences_array, batch_size, num_steps):
                global_step += 1

                feed_dict[x] = X
                feed_dict[y] = Y

                '''
                    Every PRINT_FREQUENCY-th batch is used for testing purposes only!
                    In other words,
                        - weights are not updated then
                        - current state of the graph is saved (this way, no metter when we kill the script in GPU VM, we have last previous
                            graph state saved)
                '''

                
                # Every VALIDATION_SUMMARY_FREQUENCY steps we test our network with validation batch
                # that has not been seen by the network.
                if global_step % VALIDATION_SUMMARY_FREQUENCY == 0:
                    summary_validation = sess.run(summary_op, feed_dict)
                    validation_writer.add_summary(summary_validation, global_step)
                elif global_step % TRAIN_SUMMARY_FREQUENCY == 0:
                    # Every TRAIN_SUMMARY_FREQUENCY steps we also evaluate the summary on the current training batch
                    _, summary_train = sess.run([train_step, summary_op], feed_dict)
                    train_writer.add_summary(summary_train, global_step)
                else:
                    # In every other case, just train the network
                    sess.run([train_step], feed_dict)

                # Regularly we save the current model to disk
                if global_step % CHECKPOINT_FREQUENCY == 0 :
                    if checkpoint_filename is not None:
                        saver.save(sess, "./{}-ep{}".format(checkpoint_filename, epoch), global_step=global_step)

        # At the end of the whole training process, save the model to disk
        if checkpoint_filename is not None:
            saver.save(sess, "./{}".format(checkpoint_filename), global_step=global_step)




def mainFunc(argv):
    def printUsage():
        print('main.py -n <num_cores> -x <experiment>')
        print('num_cores = Number of cores requested from the cluster. Set to -1 to leave unset')
        print('experiment = experiment setup that should be executed. Set to A, B or C')

    num_cores = -1
    num_epochs = NUM_EPOCHS
    experiment = ""
    # Command line argument handling
    try:
        opts, args = getopt.getopt(argv,"n:x:",["num_cores=", "experiment="])
    except getopt.GetoptError:
        printUsage()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            printUsage()
            sys.exit()
        elif opt in ("-n", "--num_cores"):
            num_cores = int(arg)
        elif opt in ("-x", "--experiment"):
            if arg in ("A", "B", "C"):
                experiment = arg
            else:
                printUsage()
                sys.exit(2)

    print("Executing experiment {} with {} CPU cores".format(experiment, num_cores))
    if num_cores != -1:
        # We set the op_parallelism_threads in the ConfigProto and pass it to the TensorFlow session
        configProto = tf.ConfigProto(log_device_placement=False,
                        inter_op_parallelism_threads=num_cores,
                        intra_op_parallelism_threads=num_cores)
    else:
        configProto = tf.ConfigProto(log_device_placement=False)

    print("Building graph")
    graph = None
    # Experiment C required double the default hidden state size
    state_size = CELL_SIZE
    if experiment == "C":
        state_size = 2 * CELL_SIZE

    graph = build_training_graph(state_size = state_size, downproject_cellsize = CELL_SIZE)


    sentences, index_2_word, word_2_index, _ = preprocessing.preprocess_data(   TRAINING_DATA_PATH,
                                                                                max_sentence_length=MAX_SENTENCE_LENGTH,
                                                                                vocabulary_size=VOCABULARY_SIZE)
    sentences_array = np.array(sentences)
    print("Sentences shape is {}".format(sentences_array.shape))

    print("Training network")
    # Use word2vec only for experiment B and C
    useWord2Vec = False
    if experiment in ("B", "C"):
        useWord2Vec = True
    t = time.time()
    train_network(  graph,
                    sentences_array,
                    checkpoint_filename="exp{}".format(experiment),
                    num_epochs=num_epochs,
                    configProto = configProto,
                    state_size = state_size,
                    vocabulary=word_2_index,        # used in load_embeddings method
                    useWord2Vec=useWord2Vec)        # if True, uses word2vec embedding

    print("It took {} seconds to train for {} epochs.".format(time.time() - t, num_epochs))

if __name__ == "__main__":
    mainFunc(sys.argv[1:])
