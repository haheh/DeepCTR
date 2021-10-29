# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen, weichenswc@163.com

Reference:
    [1] Cheng H T, Koc L, Harmsen J, et al. Wide & deep learning for recommender systems[C]//Proceedings of the 1st Workshop on Deep Learning for Recommender Systems. ACM, 2016: 7-10.(https://arxiv.org/pdf/1606.07792.pdf)
"""
#wide部分就是LR，逻辑回归
import tensorflow as tf

from ..feature_column import build_input_features, get_linear_logit, input_from_feature_columns
from ..layers.core import PredictionLayer, DNN
from ..layers.utils import add_func, combined_dnn_input


def WDL(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(256, 128, 64), l2_reg_linear=0.00001,
        l2_reg_embedding=0.00001, l2_reg_dnn=0, seed=1024, dnn_dropout=0, dnn_activation='relu',
        task='binary'):
    """
    :param linear_feature_columns: 输入到wide模型的特征列
    :param dnn_feature_columns: 输入到Deep模型的特征列
    :param dnn_hidden_units: list,列表里包含了DNN所有隐层的神经元数目大小，也可以是空列表，即DNN没有隐层
    :param l2_reg_linear: float. 用于wide模型的L2正则项系数
    :param l2_reg_embedding: float. 应用到嵌入向量的L2正则项系数
    :param l2_reg_dnn: float. 用于deep模型的L2正则项系数
    :param seed: integer ,用作随机种子
    :param dnn_dropout: float in [0,1), DNN的dropout概率值。防止过拟合
        dropout是一种在神经网络中进行正则化的方法，有助于减少神经元之间的相互依赖学习。
    :param dnn_activation: DNN的激活函数
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss回归损失
    :return: A Keras model instance.
    """
    """Instantiates the Wide&Deep Learning architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to wide part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """

    features = build_input_features(
        linear_feature_columns + dnn_feature_columns)#为所有的特征列构造keras tensor，结果以OrderDict形式返回

    inputs_list = list(features.values())#取OrderDict的values，即将所有特征列的值构成列表

    linear_logit = get_linear_logit(features, linear_feature_columns, seed=seed, prefix='linear',
                                    l2_reg=l2_reg_linear)#获取linear_logit（线性变换）的结果
    #为所有特征列创建嵌入矩阵，并分别返回包含SparseFeat和VarLenSparseFeat的嵌入矩阵的字典，以及包含DenseFeat的数值特征的字典
    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         l2_reg_embedding, seed)

    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)#将SparseFeat、VarLenSparseFeat和DenseFeat拼接，获得DNN的输入
    dnn_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, False, seed=seed)(dnn_input)#经过若干层隐层变换，获得DNN的输出
    dnn_logit = tf.keras.layers.Dense(
        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed))(dnn_out)#获得DNN的logit

    final_logit = add_func([dnn_logit, linear_logit])#dnn_logit和linear_logit相加

    output = PredictionLayer(task)(final_logit)#得到预测结果

    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    return model
