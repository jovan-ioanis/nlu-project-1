import numpy as np
import tensorflow as tf
import os
import sys, getopt
from graphs import MAX_SENTENCE_LENGTH, SEQUENCE_LENGTH, CELL_SIZE, build_training_graph, VOCABULARY_SIZE
from preprocessing import START_TOKEN, END_TOKEN, UNKNOWN_TOKEN, PAD_TOKEN
from main import data_iterator
import pickle

TEST_SENTENCES_PATH = "data/sentences_test"
BATCH_SIZE = 1
def load_w2i() :
    """
    Loads word to index and index to word dictionaries to convert raw text to indices.
    """
    if os.path.exists('index_2_word.pickle') and os.path.exists('word_2_index.pickle') :
        print("Reading index_2_word.pickle")
        with open('index_2_word.pickle', 'rb') as f:
            index_2_word = pickle.load(f)
        print("Reading word_2_index.pickle")
        with open('word_2_index.pickle', 'rb') as f:
            word_2_index = pickle.load(f)
        return index_2_word, word_2_index
    else :
        print("Couldn't find index_2_word.pickle and word_2_index.pickle")

def load_test_sentences(filename, max_sentence_length) :
    """
    Reads in the test sentences from text file and preprocesses them
    """
    if os.path.exists(filename) :
    	f = tf.gfile.GFile(filename, "r")
    	raw_test_data = f.read().decode("utf-8").split("\n")
    else :
        print("Couldn't find {}".format(filename))
        sys.exit()

    test_sentences = ["%s %s %s" % (START_TOKEN, x, END_TOKEN) for x in raw_test_data]
    filtered_sentences = [x.split() for x in test_sentences if len(x.split()) <= max_sentence_length]
    padded_sentences = [(x + [PAD_TOKEN] * (max_sentence_length - len(x))) for x in filtered_sentences]

    index_2_word, word_2_index = load_w2i()

    padded_sentence_indices = [[word_2_index[w] if w in word_2_index.keys() else word_2_index[UNKNOWN_TOKEN] for w in sentence] for sentence in padded_sentences]

    ## return the dicts too, don't want to have to load these twice...
    return padded_sentence_indices, index_2_word, word_2_index

def write_ppl(X, filename):
    with open("./{}".format(filename), "w") as fw:
        for i in range(len(X)):
            fw.write("{}\n".format(X[i]))

def test_perplexity(operators, checkpoint, exp, state_size):
    """
        Calculates the perplexity of the test data and outputs it to a file
        
        operators               Dictionary of graph operators to execute
        checkpoint              Filename of the checkpoint to load
        exp                     String denoting which experiment to execute, e.g. "A"
        state_size              Size of the hidden dimensionality of the cell
    """
    ## Loading in test sentences
    
    print("Loading test sentences...")
    test_sentences, index_2_word, word_2_index = load_test_sentences(TEST_SENTENCES_PATH, MAX_SENTENCE_LENGTH)
    test_sentences = np.array(test_sentences) ## array shape number of test sentences x 30 (sentence length)
    batch_size = BATCH_SIZE
    num_steps = SEQUENCE_LENGTH

    with tf.Session() as sess:
        print("Restoring graph")
        sess.run(tf.global_variables_initializer())
        # Restoring the model weights from disk
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint)
        global_step = 0

        # For the perplexity we feed zero states at the beginning of each sentence (just as in training)
        zero_state = np.zeros([BATCH_SIZE, state_size])
        feed_dict={operators['init_state_a']:zero_state, operators['init_state_b']:zero_state}
        perplexities = [] # Will contain perplexities of all sentences in the test file
        softmax_op = operators['softmax']
        x = operators['x']
        y = operators['y']

        # Iterate through the test data in batches.
        # Here we DO NOT shuffle the data
        for X, Y in data_iterator(test_sentences, batch_size, num_steps):
            global_step += 1
            
            ## preparing the input data
            feed_dict[x]= X

            # Retrieving the softmax probabilities
            softmax_list = sess.run([softmax_op], feed_dict)
            softmax = np.array(softmax_list).reshape(batch_size, num_steps, VOCABULARY_SIZE)
            
            # Perplexity calculation
            for i in range(batch_size):
                word_probs = []
                t = 0 # Index of the words in a given sentence
                # As long as we havent reached the maximum sentence length or seen the <eos>
                while t < num_steps and X[i,t] != word_2_index[END_TOKEN]:
                    label_index = Y[i,t] # The real label of the word

                    # We are interested in p(w_t|w_1,..,w_t-1)
                    # That's the probability of the correct word (we have in label_index)
                    # Softmax represents such a probability, hence we can use that
                    word_probs.append(softmax[i,t,label_index])
                    t += 1

                log_probs = np.log(word_probs)
                
                # We calculate the mean log probabilities only of the word_probs we have encountered.
                # <pad>/paddings to not participate in this mean calculation
                perp = 2**(-1.0*log_probs.mean())
                perplexities.append(perp)

            if global_step % 100 == 0:
                    print("Calculated perplexities of batch {}".format(global_step))
                    
        print("Writing ppls to file...")
        write_ppl(perplexities, "group01.perplexity" + exp)

def mainFunc(argv):
    def printUsage():
        print('test.py -n <num_cores> -x <experiment> -c <checkpoint_path')
        print('num_cores = Number of cores requested from the cluster. Set to -1 to leave unset')
        print('experiment = experiment perplexity calculation that should be executed. Set to A, B or C')
        print("checkpoint_path = Path to the checkpoint to load parameters from. e.g. './expA-500'")

    num_cores = -1
    checkpoint_filepath = ""
    experiment = ""
    # Command line argument handling
    try:
        opts, args = getopt.getopt(argv,"n:x:c:",["num_cores=", "experiment=", "checkpoint_path="])
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
        elif opt in ("-c", "--checkpoint_path"):
            if arg != "":
                checkpoint_filepath = arg
            else:
                printUsage()
                sys.exit(2)

    print("Building graph")
    graph = None
    # Experiment C required double the default hidden state size
    state_size = CELL_SIZE
    if experiment == "C":
        state_size = 2 * CELL_SIZE
    graph = build_training_graph(state_size = state_size, num_steps=29, downproject_cellsize = CELL_SIZE, batch_size = BATCH_SIZE)

    test_perplexity(graph, checkpoint_filepath, experiment, state_size)

if __name__ == "__main__":
    mainFunc(sys.argv[1:])
