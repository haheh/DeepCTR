# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen, weichenswc@163.com

Reference:
    [1] Guo H, Tang R, Ye Y, et al. Deepfm: a factorization-machine based neural network for ctr prediction[J]. arXiv preprint arXiv:1703.04247, 2017.(https://arxiv.org/abs/1703.04247)

"""
'''
学习复杂的特征交互——即低阶和高阶特征交互，不需要特征工程。
Deep FM = FM + DNN， FM模型可以抽取low-order特征，DNN可以抽取high-order特征。无需Wide&Deep模型人工特征工程。
输入仅为原始特征，而且FM和DNN共享输入特征向量，DeepFM 训练速度很快
Deep FM = FM + DNN，其实就是使用FM模型代替了Wide模型，用于提取低阶特征。
FM能反映1阶特征的重要性和2阶特征的交互。
'''

from itertools import chain

import tensorflow as tf

from ..feature_column import build_input_features, get_linear_logit, DEFAULT_GROUP_NAME, input_from_feature_columns
from ..layers.core import PredictionLayer, DNN
from ..layers.interaction import FM
from ..layers.utils import concat_func, add_func, combined_dnn_input

#对比wide&deep的代码，可以看出两者实现上差别并不大，仅多了一个FM模型
def DeepFM(linear_feature_columns, dnn_feature_columns, fm_group=(DEFAULT_GROUP_NAME,), dnn_hidden_units=(256, 128, 64),
           l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, seed=1024, dnn_dropout=0,
           dnn_activation='relu', dnn_use_bn=False, task='binary'):
    """Instantiates the DeepFM Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param fm_group: list, group_name of features that will be used to do feature interactions相互作用.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
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

    features = build_input_features(
        linear_feature_columns + dnn_feature_columns)#为所有的特征列构造keras tensor，结果以OrderDict形式返回

    inputs_list = list(features.values())#取OrderDict的values，即将所有特征列的值构成列表

    linear_logit = get_linear_logit(features, linear_feature_columns, seed=seed, prefix='linear',
                                    l2_reg=l2_reg_linear)#获取linear_logit（线性变换）的结果
#为所有特征列创建嵌入矩阵，并分别返回包含SparseFeat和VarLenSparseFeat的嵌入矩阵的字典，以及包含DenseFeat的数值特征的字典
    group_embedding_dict, dense_value_list = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding,
                                                                        seed, support_group=True)
#输入SparseFeat、VarLenSparseFeat的嵌入到FM模型，得到logit
    fm_logit = add_func([FM()(concat_func(v, axis=1))
                         for k, v in group_embedding_dict.items() if k in fm_group])

    dnn_input = combined_dnn_input(list(chain.from_iterable(
        group_embedding_dict.values())), dense_value_list)#将SparseFeat、VarLenSparseFeat和DenseFeat拼接，获得DNN的输入
    dnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(dnn_input)#经过若干层隐层变换，获得DNN的输出
    dnn_logit = tf.keras.layers.Dense(
        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=seed))(dnn_output)#获得DNN的logit

    final_logit = add_func([linear_logit, fm_logit, dnn_logit])#linear_logit、fm_logit和dnn_logit相加

    output = PredictionLayer(task)(final_logit)#得到预测结果
    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    return model
