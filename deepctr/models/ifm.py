# -*- coding:utf-8 -*-
"""
Author:
    zanshuxun, zanshuxun@aliyun.com
Reference:
    [1] Yu Y, Wang Z, Yuan B. An Input-aware Factorization Machine for Sparse Prediction[C]//IJCAI. 2019: 1466-1472.
    (https://www.ijcai.org/Proceedings/2019/0203.pdf)
"""

import tensorflow as tf
from tensorflow.python.keras.layers import Lambda

from ..feature_column import build_input_features, get_linear_logit, input_from_feature_columns, SparseFeat, \
    VarLenSparseFeat
from ..layers.core import PredictionLayer, DNN
from ..layers.interaction import FM
from ..layers.utils import concat_func, add_func, combined_dnn_input, softmax


def IFM(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(256, 128, 64),
        l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, seed=1024, dnn_dropout=0,
        dnn_activation='relu', dnn_use_bn=False, task='binary'):
    """Instantiates the IFM Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN 层数和单元
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """

    if not len(dnn_hidden_units) > 0:
        raise ValueError("dnn_hidden_units is null!")

    features = build_input_features(
        linear_feature_columns + dnn_feature_columns)#为所有的特征列构造keras tensor，结果以OrderDict形式返回

    sparse_feat_num = len(list(filter(lambda x: isinstance(x, SparseFeat) or isinstance(x, VarLenSparseFeat),
                                      dnn_feature_columns)))
    #filter(function, iterable)返回可迭代对象，用list()构建列表。其中iterable序列的每个元素作为参数传递给function()进行判断，
    # 然后返回 True 或 False，最后将返回 True 的元素放到新interable中。
    # 此处lambda()作为function()。
    # isinstance(object, classinfo) object：实例对象。classinfo：可以是直接或间接类名、基本类型或者由它们组成的元组。
        # 返回值：如果对象的类型与参数二的类型（classinfo）相同则返回 True，否则返回 False。
    #此处dnn_feature_columns的‘元素’依次传给lambda()，在lambda()里赋值给x，isinstance()判断是否是相应类型。
    inputs_list = list(features.values())

    sparse_embedding_list, _ = input_from_feature_columns(features, dnn_feature_columns,
                                                          l2_reg_embedding, seed)
    if not len(sparse_embedding_list) > 0:
        raise ValueError("there are no sparse features")

    dnn_input = combined_dnn_input(sparse_embedding_list, [])
    dnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(dnn_input)
    # here, dnn_output is the m'_{x}
    dnn_output = tf.keras.layers.Dense(#获得DNN的logit（在wdl.py中）
        sparse_feat_num, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=seed))(dnn_output)
    # input_aware_factor m_{x,i}
    input_aware_factor = Lambda(lambda x: tf.cast(tf.shape(x)[-1], tf.float32) * softmax(x, dim=1))(dnn_output)

    linear_logit = get_linear_logit(features, linear_feature_columns, seed=seed, prefix='linear',
                                    l2_reg=l2_reg_linear, sparse_feat_refine_weight=input_aware_factor)

    fm_input = concat_func(sparse_embedding_list, axis=1)
    refined_fm_input = Lambda(lambda x: x[0] * tf.expand_dims(x[1], axis=-1))(
        [fm_input, input_aware_factor])
    fm_logit = FM()(refined_fm_input)

    final_logit = add_func([linear_logit, fm_logit])

    output = PredictionLayer(task)(final_logit)
    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    return model
