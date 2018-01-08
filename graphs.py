import tensorflow as tf

VOCABULARY_SIZE = 20000
CELL_SIZE = 512
BATCH_SIZE = 64
MAX_SENTENCE_LENGTH = 30
LEARNING_RATE = 1e-4
WORD_EMBEDDING_SIZE = 100
SEQUENCE_LENGTH = MAX_SENTENCE_LENGTH-1 # Network used a step of 1 so that inputs and outputs can match up, hence num_steps = MAX_SENTENCE_LENGTH-1

def build_training_graph(
    state_size = CELL_SIZE,
    downproject_cellsize = CELL_SIZE,
    vocabulary_size = VOCABULARY_SIZE,
    batch_size = BATCH_SIZE,
    num_steps = SEQUENCE_LENGTH,
    learning_rate = LEARNING_RATE,
    word_embedding_size = WORD_EMBEDDING_SIZE
    ):
    """
        Builds the TensorFlow graph

        state_size              Size of the hidden state in an RNN cell
        downproject_cellsize    Size of the downprojection step before prediction.
        vocabulary_size         Size of our vocabulary
        batch_size              Number of sentences per batch
        num_steps               Number of steps of our RNN / Sequence length
        learning_rate           Learning rate for the AdamOptimizer
        word_embedding_size     Size of the dimensionality of the word embeddings
    """

    tf.reset_default_graph() # Always start with a fresh graph

    ###
    # Placeholders for data and hidden states
    ###
    x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder') # [batch_size, num_steps]
    y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder') # [batch_size, num_steps]

    init_state_a = tf.placeholder(tf.float32, [batch_size, state_size], name='input_state_a') # [batch_size, num_steps]
    init_state_b = tf.placeholder(tf.float32, [batch_size, state_size], name='input_state_b') # [batch_size, num_steps]

    init_state = (init_state_a, init_state_b)

    ###
    # Word embedding layer
    ###
    with tf.variable_scope("word_embedding"):
        W_embed = tf.get_variable(name="W_embed", initializer=tf.random_uniform([vocabulary_size, word_embedding_size], -1.0, 1.0)) # [vocabulary_size, word_embedding_size]
        embeddings = tf.nn.embedding_lookup(W_embed, x) # [batch_size, num_steps, word_embedding_size]

    ###
    # Creating the RNN by unrolling the individual cells
    ###
    with tf.variable_scope("rnn"):
        current_state = init_state # [batch_size, state_size]
        rnn_outputs = []

        cell = tf.contrib.rnn.BasicLSTMCell(num_units=state_size, state_is_tuple=True)

        for i in range(num_steps):
            if i > 0 :
                tf.get_variable_scope().reuse_variables() # Reuse the variables of the cells

            output, current_state = cell(inputs=embeddings[:, i, :], state=current_state)

            rnn_outputs.append(output)
        final_state = rnn_outputs[-1]

    rnn_outputs_3d = tf.stack(rnn_outputs, axis=1)
    rnn_outputs_2d = tf.reshape(rnn_outputs_3d, [batch_size * num_steps, state_size])

    ###
    # Projecting down if the hidden layer size is not equal to the downproject size
    ###
    if state_size != downproject_cellsize :
        print("Adding Downprojection step: Hidden state size is {}, but downprojection size is {}".format(state_size, downproject_cellsize))
        with tf.variable_scope('downprojection'):
            projection = tf.get_variable('projection_weights', [state_size, downproject_cellsize], initializer=tf.contrib.layers.xavier_initializer())
            rnn_outputs_2d = tf.matmul(rnn_outputs_2d, projection)

    ###
    # Weights for the softmax
    ###
    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [downproject_cellsize, vocabulary_size], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b', [vocabulary_size], initializer=tf.contrib.layers.xavier_initializer())

    logits = tf.matmul(rnn_outputs_2d, W) + b
    softmax = tf.nn.softmax(logits) # For perplexity calculation
    prediction_indices = tf.argmax(logits, axis=1) # For each step and every sentence we find the word with the highest logit
    

    # We use the AdamOptimizer with gradient clipping as requested
    unrolled_labels = tf.reshape(y, [batch_size * num_steps])
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=unrolled_labels, logits=logits)
    total_loss = tf.reduce_sum(losses)
    
    optimizer = tf.train.AdamOptimizer(learning_rate)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(total_loss, tvars), clip_norm=10)
    train_step = optimizer.apply_gradients(zip(grads, tvars))

    ###
    # Creating summary for tensorboard.
    ###
    accuracy_equals = tf.equal(unrolled_labels, tf.cast(prediction_indices, tf.int32)) # [batch_size * num_steps] tensor of {True,False} values
    accuracy = tf.reduce_mean(tf.cast(accuracy_equals, tf.float32))
    tf.summary.scalar("total_loss", total_loss)
    tf.summary.scalar("accuracy", accuracy)
    # merge all summaries into a single "operation" which we can execute in a session
    summary_op = tf.summary.merge_all()

    return dict(
        x = x,
        y = y,
        train_step = train_step,
        summary_op = summary_op,
        softmax = softmax, ## for perplexity calculation
        predictions = prediction_indices,
        init_state_a = init_state_a,
        init_state_b = init_state_b,
        final_state = current_state
    )