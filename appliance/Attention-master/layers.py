from inits import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

def reset_layer_uid():
    """Helper function, reset unique layer IDS"""
    for key, val in _LAYER_UIDS.items():
        _LAYER_UIDS[key] = 0
    # _LAYER_UIDS = {}




def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        __call__(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for __call__()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def __call__(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self.__call__(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def __call__(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class BatchNorm(Layer):
    """Batch Normalization layer."""
    def __init__(self, input_dim, dropout, placeholders, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        with tf.variable_scope(self.name + '_vars'):
            self.vars['gamma'] = gamma([input_dim], name='gamma')
            self.vars['beta'] = zeros([input_dim], name='beta')

        if self.logging:
            self._log_vars()

    def __call__(self, inputs):
        x = inputs

        mean, variance = tf.nn.moments(x, axes=0)
        output = tf.nn.batch_normalization(x, mean, variance, self.vars['beta'], self.vars['gamma'], 0.00001)

        return tf.nn.dropout(output, 1-self.dropout)


class LSTM(Layer):
    """LSTM layer."""
    def __init__(self, input_dim, hidden_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False, **kwargs):
        super(LSTM, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.bias = bias

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights_wx'] = glorot([input_dim, hidden_dim*4],
                                          name='weights_wx')
            self.vars['weights_wh'] = glorot([hidden_dim, hidden_dim*4],
                                          name='weights_wh')
            if self.bias:
                self.vars['bias'] = zeros([hidden_dim*4], name='bias')


        if self.logging:
            self._log_vars()

    def __call__(self, inputs, states):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        c, h = tf.split(axis=1, num_or_size_splits=2, value=states)

        z = dot(x, self.vars['weights_wx'], sparse=self.sparse_inputs) + dot(h, self.vars['weights_wh'], sparse=self.sparse_inputs)
        # bias
        if self.bias:
            z += self.vars['bias']
        # LSTM
        i, f, o, u = tf.split(axis=1, num_or_size_splits=4, value=z)
        i = tf.nn.sigmoid(i)
        f = tf.nn.sigmoid(f)
        o = tf.nn.sigmoid(o)
        u = tf.tanh(u)
        c = f*c + i*u
        h = o*tf.tanh(c)
        state = tf.concat([c, h], axis=1)

        # transform
        output = self.act(h)

        return output, state


class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def __call__(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class GraphConvolution_v2(Layer):
    """Graph convolution layer with different variables for ego entries.
       There are two fusion_method: ['concat', 'add']
    """
    def __init__(self, input_dim, output_dim, placeholders, fusion_method='add', dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution_v2, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.support_diag = [tf.sparse.eye(placeholders['batch_size']) for support in self.support]
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        assert fusion_method in ['concat', 'add']
        self.fusion_method = fusion_method

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                if self.fusion_method == 'add':
                    self.vars['weights_s_' + str(i)] = glorot([input_dim, output_dim],
                                                            name='weights_s_' + str(i))

                    self.vars['weights_e_' + str(i)] = glorot([input_dim, output_dim],
                                                            name='weights_e_' + str(i))

                elif self.fusion_method == 'concat':
                    self.vars['weights_s_' + str(i)] = glorot([input_dim, output_dim/2],
                                                            name='weights_s_' + str(i))

                    self.vars['weights_e_' + str(i)] = glorot([input_dim, output_dim/2],
                                                            name='weights_e_' + str(i))

            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def __call__(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup_s = dot(x, self.vars['weights_s_' + str(i)],
                              sparse=self.sparse_inputs)
                pre_sup_e = dot(x, self.vars['weights_e_' + str(i)],
                            sparse=self.sparse_inputs)
            else:
                pre_sup_s = self.vars['weights_s_' + str(i)]
                pre_sup_e = self.vars['weights_e_' + str(i)]
            support_s = dot(self.support[i], pre_sup_s, sparse=True)
            support_e = dot(self.support_diag[i], pre_sup_e, sparse=True)
            if self.fusion_method == 'add':
                support = support_s + support_e
            else:
                support = tf.concat([support_s, support_e], axis=1)
            supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class GraphConvolution_relative(Layer):
    """Graph convolution layer."""
    def __init__(self, output_dim, placeholders, dropout=0.,
                 sparse_inputs=True, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution_relative, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([1, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def __call__(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = tf.sparse.to_dense(x)

        x = tf.abs(tf.transpose(x) - x * (-1))
        x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                support_feature = dot(self.support[i], x, sparse=True)
                support_feature = tf.expand_dims(tf.diag_part(support_feature), axis=1)
                support = dot(support_feature, self.vars['weights_' + str(i)])
            else:
                raise ValueError('Does not support featureless')
            supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)
