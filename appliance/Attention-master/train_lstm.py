def get_flags(model, learning_rate, epochs, hidden1, embedding, hidden2, hidden3, grads_norm, dropout, weight_decay, early_stopping, batch_norm, grads_clip):
    FLAGS = flags.FLAGS
    flags.DEFINE_string('model', model, 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
    flags.DEFINE_float('learning_rate', learning_rate, 'Initial learning rate, choosing from [0.01, 0.005, 0.001, 0.0005].')
    flags.DEFINE_integer('epochs', epochs, 'Number of epochs to train.')
    flags.DEFINE_integer('hidden1', hidden1, 'Number of units in hidden layer 1.')
    flags.DEFINE_integer('embedding', embedding, 'Number of units in embedings.')
    flags.DEFINE_integer('hidden2', hidden2, 'Number of units in LSTM hidden layer.')
    flags.DEFINE_integer('hidden3', hidden3, 'Number of units before DMN.')
    flags.DEFINE_float('grads_norm', grads_norm, 'clip gradients.')
    flags.DEFINE_boolean('grads_clip', grads_clip, 'Whether conduct gradients clipping.')
    flags.DEFINE_float('dropout', dropout, 'Dropout rate (1 - keep probability).')
    flags.DEFINE_float('weight_decay', weight_decay, 'Weight for L2 loss on embedding matrix.')
    flags.DEFINE_integer('early_stopping', early_stopping, 'Tolerance for early stopping (# of epochs).')
    flags.DEFINE_boolean('batch_norm', batch_norm, 'Whether conduct batch normalization.')

    print('\n')
    print('Preparing parameters for this run:')
    print('{ \'model\': %s,'%FLAGS.model)
    print('  \'learning_rate\': %s,'%FLAGS.learning_rate)
    print('  \'epochs\': %s,'%FLAGS.epochs)
    print('  \'hidden1\': %s,'%FLAGS.hidden1)
    print('  \'embedding\': %s,'%FLAGS.embedding)
    print('  \'hidden2\': %s,'%FLAGS.hidden2)
    print('  \'hidden3\': %s,'%FLAGS.hidden3)
    print('  \'grads_norm\': %s,'%FLAGS.grads_norm)
    print('  \'grads_clip\': %s,'%FLAGS.grads_clip)
    print('  \'dropout\': %s,'%FLAGS.dropout)
    print('  \'weight_decay\': %s,'%FLAGS.weight_decay)
    print('  \'early_stopping\': %s,'%FLAGS.early_stopping)
    print('  \'batch_norm\': %s}'%FLAGS.batch_norm)
    print('\n')

    return FLAGS




def prepare_model(FLAGS):
    frames = extract_frames()
    data_l = data_loader(frames[0])
    adj_conv, _, _, _, features, _, _, _, _, _ = next(data_l)
    features = preprocess_features(features)

    if FLAGS.model == 'dense_lstm':
        num_supports = 1
        model_func = MLP_LSTM

    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))
    # Define placeholders
    feature_input_ph = {
        'features': tf.placeholder(tf.float32, shape=[None, 3],name='features')),
        'labels': tf.placeholder(tf.float32, shape=[None, 2], name='labels'),
        # 'labels_mask': tf.placeholder(tf.float32, shape=(None, 1)),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'grads_norm':tf.placeholder_with_default(FLAGS.grads_norm, shape=()),
        'input_dim': 3,
        'states': tf.placeholder(tf.float32, shape=(None, 2*FLAGS.hidden2))
    }
    placeholders = feature_input_ph

    # Create model

    model = model_func(placeholders, features[2][1], name = FLAGS.model, logging=False, BN=FLAGS.batch_norm, grads_clip=FLAGS.grads_clip)
    return model, placeholders



def lstm_train(model, placeholders, FLAGS):
    # Initialize session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    # Init variables
    sess.run(tf.global_variables_initializer())
    best_mse = None
    # Train model
    epochs = FLAGS.epochs
    frames = extract_frames()

    for epoch in range(epochs):
        train_loss = 0.
        count = 0.
        test_loss = 0.
        overall_t = tqdm(frames, desc="Training Epoch %d" %epoch)
        for init_frame in overall_t:
            # train_loader = data_loader(init_frame)
            data_gen = iterate_data()
            # Training epochs
            # t_train = tqdm(range(train_loader.num), desc="Training trajectories starting from frame %d" %init_frame)

            for ele in data_gen:
                source_data, source_len, taget_data, target_len = ele
                states = np.zeros((1, 2*FLAGS.hidden2))
                for idx in range(source_len):
                    if idx == 0:
                        continue
                    features_i = sparse_to_tuple(features)
                    feed_dict = construct_feed_dict(features_i, [adj], y, train_mask, placeholders)
                    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
                    feed_dict.update({placeholders['states']: states})
                    states = sess.run(model.states, feed_dict=feed_dict)
                for idx in range(target_len):
                    if idx == 0:
                        continue
                    # Training step
                    count += 1
                    outs = sess.run([model.states, model.loss, model.opt_op], feed_dict=feed_dict)
                    states = outs[0]
                    train_loss += outs[1]

        # overall_test = tqdm(frames, desc="Testing Epoch %d" %epoch)
        #
        # count = 0.
        # for init_frame in overall_test:
        #     test_loader = data_loader(init_frame)
        #
        #
        #     for idx in range(test_loader.num):
        #         adj, _, _, _, features, y, _, test_mask, _, _ = next(test_loader)
        #         if np.sum(test_mask) == 0:
        #             break
        #         features_i = sparse_to_tuple(features)
        #         feed_dict = construct_feed_dict(features_i, [adj], y, test_mask, placeholders)
        #         if idx == 0:
        #             states = np.zeros((1, 2*FLAGS.hidden2))
        #         feed_dict.update({placeholders['states']: states})
        #
        #         # No action in the first 2 frames
        #         if idx < 20:
        #             states = sess.run(model.states, feed_dict=feed_dict)
        #         else:
        #             count += 1
        #             # Training step
        #             outs = sess.run([model.states, model.accuracy], feed_dict=feed_dict)
        #             states = outs[0]
        #             test_loss += np.mean(outs[1])
        #             overall_test.set_postfix(ordered_dict={'mse': test_loss/count})

        # if best_mse == None:
        #     best_mse = test_loss/count
        #     model.save_v2(sess)
        # else:
        #     if best_mse > (test_loss/count):
        #         best_mse = test_loss/count
        #         model.save_v2(sess)
