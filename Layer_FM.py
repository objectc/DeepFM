import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import GlorotNormal


class Layer_FM(Layer):
    def __init__(self, sp_features_len, num_features_len, embedding_size, **kwargs):
        super(Layer_FM, self).__init__(**kwargs)
        self._sp_features_len = sp_features_len
        self._num_features_len = num_features_len
        self._embedding_size = embedding_size
        self._FM_V = None
        self._idx_numeric = None

    def init_embedding(self, feature_size):
        if not self._FM_V:
            self._FM_V = self.add_weight(shape=(feature_size, self._embedding_size),
                                         initializer=GlorotNormal(),
                                         dtype=tf.dtypes.float32,
                                         trainable=True, name='FM_V')

    def call(self, inputs):
        sp_inputs = inputs['sparse']
        num_inputs = inputs['numeric']
        # use tf.shape to find the runtime batch size
        batch_size = tf.shape(sp_inputs)[0]
        sp_feature_size = sp_inputs.shape[1]
        num_feature_size = num_inputs.shape[1]

        feature_size = sp_feature_size + num_feature_size
        self.init_embedding(feature_size)
        # 'non_zero_idx_sparse' shape (batch * sp_feature_size, 2),
        # the second dim "2" represents the coordinates of the elements(#batch, #feature)
        non_zero_idx_sparse = tf.where(tf.not_equal(inputs['sparse'], 0))
        # get rid of the #batch and reshape it
        idx_sparse = tf.reshape(
            non_zero_idx_sparse[:, 1], shape=(-1, self._sp_features_len))
        feat_vals_sparse = tf.gather_nd(inputs['sparse'], non_zero_idx_sparse)
        feat_vals_sparse = tf.reshape(
            feat_vals_sparse, shape=(-1, self._sp_features_len))

        # numeric features are call considered as non zero idx
        # 'self.numeric_idx' should only generate once for all the dataset
        if not self._idx_numeric:
            # id start from sp_feature_size
            numeric_idx_one_batch = tf.range(
                sp_feature_size, sp_feature_size+num_feature_size, dtype=tf.int64)
            # tile to fit the batch
            numeric_idx_one_batch = tf.expand_dims(
                numeric_idx_one_batch, axis=0)
            self._idx_numeric = tf.tile(numeric_idx_one_batch, (batch_size, 1))
        idx = tf.concat([idx_sparse, self._idx_numeric], 1)
        feat_vals = tf.concat([feat_vals_sparse, inputs['numeric']], axis=1)
        feat_vals = tf.reshape(
            feat_vals,
            shape=(-1, self._sp_features_len+self._num_features_len, 1))
        embeddings = tf.nn.embedding_lookup(self._FM_V, idx)
        v_em = tf.multiply(embeddings, feat_vals, name='v_em')
        sum_square = tf.square(tf.reduce_sum(v_em, 1))
        square_sum = tf.reduce_sum(tf.square(v_em), 1)
        y_v = 0.5 * tf.reduce_sum(tf.subtract(sum_square, square_sum), 1)
        # outputs = {"second_order": y_v, "v_em": v_em}
        outputs = (y_v, v_em)
        return outputs
