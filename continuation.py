import numpy as np
import tensorflow as tf
import os
import sys, getopt
import preprocessing
from graphs import build_training_graph, VOCABULARY_SIZE, CELL_SIZE, BATCH_SIZE, LEARNING_RATE, WORD_EMBEDDING_SIZE, SEQUENCE_LENGTH
from preprocessing import START_TOKEN, END_TOKEN, UNKNOWN_TOKEN, PAD_TOKEN
import pickle


CONTINUATION_SENTENCES_PATH = "data/sentences.continuation"
MAX_SENTENCE_LENGTH = 20 # For the continuation task we limit the length of the sentences to predict
BATCH_SIZE = 1
NUM_STEPS = 1
GROUP_NAME = '01'

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
    else :
        print("Couldn't find index_2_word.pickle and word_2_index.pickle")

    return index_2_word, word_2_index

def load_sentence_beginnings(filename) :
    """
    Reads in the provided sentence beginnings we should use for continuation
    """
    if os.path.exists(filename) :
        f = tf.gfile.GFile(filename, "r")
        raw_data = f.read().decode("utf-8").split("\n")
    else :
        print("Couldn't find {}".format(filename))

    beginnings_padded = ["%s %s" % (START_TOKEN, x) for x in raw_data]
    filtered_sentences = [x.split() for x in beginnings_padded if len(x.split()) < MAX_SENTENCE_LENGTH]
    index_2_word, word_2_index = load_w2i()
    beginning_indices = [[word_2_index[w] if w in word_2_index.keys() else word_2_index[UNKNOWN_TOKEN] for w in sentence] for sentence in filtered_sentences]
    return beginning_indices, index_2_word, word_2_index, filtered_sentences

def write_submission(X, filename = 'submission'):
    with open("./{}".format(filename), "w") as fw:
        for i in range(len(X)):
            print(len(X[i]))
            for j in range(len(X[i])):
                if j == 0 :
                    continue
                fw.write("{} ".format(X[i][j]))
            fw.write("\n")

def continue_sentences(operators, checkpoint, state_size) :
    """
        Load sentence beginnings, continues sentences up to the max length and writes the result back to disk
        
        operators               Dictionary of graph operators to execute
        checkpoint              Filename of the checkpoint to load
        state_size              Size of the hidden dimensionality of the cell
    """

    # Load the sentences beginnings from disk
    beginnings, index_2_word, word_2_index, filtered_sentences = load_sentence_beginnings(CONTINUATION_SENTENCES_PATH)
    beginnings = np.array(beginnings)

    zero_state = (np.zeros((BATCH_SIZE, state_size)),)*2
    with tf.Session() as sess:
        print("Restoring graph")
        sess.run(tf.global_variables_initializer())
        # Restoring the model weights from disk
        saver = tf.train.Saver(max_to_keep=5)
        saver.restore(sess, checkpoint)

        total_full_output = []

        global_step = 0

        for i in range(len(beginnings)) : # For each sentence
            global_step += 1
            step = 0 # Step within the sentence
            last_prediction = None
            full_sentence = []

            if global_step % 100 == 0:
                print("Continuing sentence {}".format(i))
            # For each sentence we start with a zero hidden state
            state = zero_state

            # We then feed the sentence beginnings into the network word by word
            for j in range(len(beginnings[i])) :
                
                x = np.array(beginnings[i][j]).reshape((1,1))
                
                feed_dict = { operators['x']: x }
                # Initially we feed the zero_state, but after that we feed the state of the previous iteration
                feed_dict[operators['init_state_a']] = state[0]
                feed_dict[operators['init_state_b']] = state[1]

                step += 1

                if x[0,0] == word_2_index[UNKNOWN_TOKEN] :
                    full_sentence.append(filtered_sentences[i][j])
                else :
                    full_sentence.append(index_2_word[x[0,0]])

                # Run the network for one word
                last_prediction, state = sess.run([operators['predictions'], operators['final_state']], feed_dict)
                
                last_prediction = np.array(last_prediction).reshape((1,1))

            full_sentence.append(index_2_word[last_prediction[0,0]])

            # We finished feeding the sentence beginnings.Now we use the derived output state
            # and feed it to predict the next words.
            for j in range(MAX_SENTENCE_LENGTH - step) :
                
                # Break when we predicted <eos>
                if last_prediction[0,0] == word_2_index[END_TOKEN] :
                    break

                # Again - Reuse the state from the previous step
                feed_dict = { operators['x']: last_prediction }
                feed_dict[operators['init_state_a']] = state[0]
                feed_dict[operators['init_state_b']] = state[1]

                # We predict the next word and also keep the hidden state
                last_prediction, state = sess.run([operators['predictions'], operators['final_state']], feed_dict)
                last_prediction = np.array(last_prediction).reshape((1,1))
                full_sentence.append(index_2_word[last_prediction[0,0]])

            total_full_output.append(full_sentence)

        write_submission(total_full_output, filename="group" + GROUP_NAME + ".continuation")

def mainFunc(argv):
    def printUsage():
        print('test.py -x <experiment> -c <checkpoint_path')
        print('experiment = experiment sentence completion that should be executed. Set to A, B or C')
        print("checkpoint_path = Path to the checkpoint to load parameters from. e.g. './expA-500'")

    checkpoint_filepath = ""
    experiment = ""
    # Command line argument handling
    try:
        opts, args = getopt.getopt(argv,"x:c:",["experiment=", "checkpoint_path="])
    except getopt.GetoptError:
        printUsage()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            printUsage()
            sys.exit()
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

    print("Building graph!")
    graph = None
    # Experiment C required double the default hidden state size
    state_size = CELL_SIZE
    if experiment == "C":
        state_size = 2 * CELL_SIZE

    # Create the graph with BATCH_SIZE and NUM_STEPS set to 1,
    # because we want to predict and feed states one-by-one
    graph = build_training_graph(state_size = state_size, downproject_cellsize = CELL_SIZE, batch_size=BATCH_SIZE, num_steps=NUM_STEPS)


    continue_sentences(graph, checkpoint_filepath, state_size)

if __name__ == "__main__":
    mainFunc(sys.argv[1:])
