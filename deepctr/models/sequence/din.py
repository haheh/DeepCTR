# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen, weichenswc@163.com

Reference:
    [1] Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068. (https://arxiv.org/pdf/1706.06978.pdf)
"""
'''
embedding&MLP模型通过SUM pooling汇聚用户行为特征组的所有嵌入向量，会得到固定长度的、能反映用户兴趣的表示向量
使用固定长度的向量将成为这类模型的瓶颈，因为通过embedding&MLP的方式很难从丰富的历史行为中捕捉到用户的多元化兴趣。
提高嵌入向量的维度(1)会增加学习的参数数量;(2)会导致过拟合
由于用户在进入显示广告的系统时并没有明确表达其意图，因此在建立CTR预测模型时，需要有效的方法从丰富的历史行为中提取用户的兴趣。
因此，DIN设计了一个局部激活单元并协同确定好的广告从用户历史行为中自适应地学习用户兴趣表示。
这种表示向量随着投放广告的不同而不同，极大的提高了模型的表达能力。
相比embedding&MLP模型，DIN在Sum Pooling前加了一个激活单元。激活单元输入是embedding&MLP模型得到的原始用户兴趣表示向量和候选广告表示向量，
输出一个激活权重值。该权重值反映了历史行为与候选广告的相关性，不同的广告算出来的值不同。接着通过有权重的sum pooling来得到自适应的计算用户表示

从激活单元的架构图可以看出，DIN的注意力机制和传统的注意力机制还是有很大不同的，传统的注意力机制会对两个输入u和v做点积u·v或者经过一个简单的前向神经网络u·W·v得到最终的权重值。
但是DIN的局部激活单元在得到u和的v外积后，会和原始的u和v拼接起来，再经过一个全连接层得到最终的权重值，这样做的原因是为了帮助相关性建模。
传统的attention还会做一个归一化操作，即令所有w相加等于1，目的是为了保留用户的兴趣强度。但是在DIN模型没有这个限制条件。
相反，所有权重w之和的值在某些程度上近似于用户兴趣的激活强度。举个例子，如果一个人历史行为中包含了90%的衣服和10%的电子产品，给定两个候选广告：T恤和手机，T
恤会激活了大部分属于衣服的历史行为，其weight值比手机更大（更强烈的兴趣）。假设T恤得到的w1为0.8，手机得到的w2为0.4， 而根据传统的attention方式，正则化之后w1=0.67，w2=0.33，
用户喜欢的强度反而变弱了，损失了数值上的分辨率。
'''
import tensorflow as tf

from ...feature_column import SparseFeat, VarLenSparseFeat, DenseFeat, build_input_features
from ...inputs import create_embedding_matrix, embedding_lookup, get_dense_input, varlen_embedding_lookup, \
    get_varlen_pooling_list
from ...layers.core import DNN, PredictionLayer
from ...layers.sequence import AttentionSequencePoolingLayer
from ...layers.utils import concat_func, NoMask, combined_dnn_input


def DIN(dnn_feature_columns, history_feature_list, dnn_use_bn=False,
        dnn_hidden_units=(256, 128, 64), dnn_activation='relu', att_hidden_size=(80, 40), att_activation="dice",
        att_weight_normalization=False, l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0, seed=1024,
        task='binary'):
    """Instantiates the Deep Interest Network architecture.

    :param dnn_feature_columns: 输入到DNN的特征列，一般会包括SparseFeat，VarLenSparseFeat和DenseFeat.
    :param history_feature_list: list,用于指定历史行为序列特征的名字
    :param dnn_use_bn: bool. 在激活函数之前是否使用BatchNormalization
    :param dnn_hidden_units: list,包含神经网络隐层单元大小的列表
    :param dnn_activation: 神经网络使用的激活函数
    :param att_hidden_size: list,正整数列表, 包含注意力网络的每层的单元数
    :param att_activation: 注意力网络的激活函数
    :param att_weight_normalization: bool.是否在执行注意力激活函数前对注意力分数做归一化
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.

    """
    """Instantiates the Deep Interest Network architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param history_feature_list: list,to indicate  sequence sparse field
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param dnn_activation: Activation function to use in deep net
    :param att_hidden_size: list,list of positive integer , the layer number and units in each layer of attention net
    :param att_activation: Activation function to use in attention net
    :param att_weight_normalization: bool.Whether normalize the attention score of local activation unit.
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.

    """

    features = build_input_features(dnn_feature_columns)#为dnn_feature_columns构造keras tensor，结果以OrderDict形式返回
    #分别过滤出SparseFeat、DenseFeat、VarLenSparseFeat
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)) if dnn_feature_columns else []
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeat), dnn_feature_columns)) if dnn_feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), dnn_feature_columns)) if dnn_feature_columns else []

    history_feature_columns = []
    sparse_varlen_feature_columns = []
    history_fc_names = list(map(lambda x: "hist_" + x, history_feature_list))#将history_feature_list里的名字都加上前缀'hist_'
    #将varlen_sparse_feature_columns中属于history_feature_columns的特征列找出来，归入history_feature_columns，
        # 剩下的归入sparse_varlen_feature_columns，因此我们在构造历史行为特征列时，命名前缀要加上'hist_'
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        if feature_name in history_fc_names:#判断是否属于历史行为特征列
            history_feature_columns.append(fc)
        else:
            sparse_varlen_feature_columns.append(fc)

    inputs_list = list(features.values())#取OrderDict的values，即将所有特征列的值构成列表
    #从所有特征列中筛选出SparseFeat和VarLenSparseFeat，然后调用inputs.py中的函数create_embedding_dict()为筛选的特征列创建嵌入矩阵
    embedding_dict = create_embedding_matrix(dnn_feature_columns, l2_reg_embedding, seed, prefix="")
    #从features中查询history_feature_list的嵌入矩阵，以字典形式返回查询结果
    query_emb_list = embedding_lookup(embedding_dict, features, sparse_feature_columns, history_feature_list,
                                      history_feature_list, to_list=True)
    #从features中查询history_fc_names的嵌入矩阵，以字典形式返回查询结果
    keys_emb_list = embedding_lookup(embedding_dict, features, history_feature_columns, history_fc_names,
                                     history_fc_names, to_list=True)
    #查询sparse_feature_columns的嵌入矩阵，以字典形式返回查询结果
    dnn_input_emb_list = embedding_lookup(embedding_dict, features, sparse_feature_columns,
                                          mask_feat_list=history_feature_list, to_list=True)
    #从所有特征列中选出DenseFeat，并以列表形式返回结果
    dense_value_list = get_dense_input(features, dense_feature_columns)
    #获取varlen_sparse_feature_columns的嵌入矩阵
    sequence_embed_dict = varlen_embedding_lookup(embedding_dict, features, sparse_varlen_feature_columns)
    #获取varlen_sparse_feature_columns池化后的嵌入向量
    sequence_embed_list = get_varlen_pooling_list(sequence_embed_dict, features, sparse_varlen_feature_columns,
                                                  to_list=True)

    dnn_input_emb_list += sequence_embed_list

    keys_emb = concat_func(keys_emb_list, mask=True)
    deep_input_emb = concat_func(dnn_input_emb_list)
    query_emb = concat_func(query_emb_list, mask=True)
    #应用注意力机制，得到加权后的结果
    hist = AttentionSequencePoolingLayer(att_hidden_size, att_activation,
                                         weight_normalization=att_weight_normalization, supports_masking=True)([
        query_emb, keys_emb])

    deep_input_emb = tf.keras.layers.Concatenate()([NoMask()(deep_input_emb), hist])#将deep_input_emb和hist拼接
    deep_input_emb = tf.keras.layers.Flatten()(deep_input_emb)#展平
    dnn_input = combined_dnn_input([deep_input_emb], dense_value_list)#拼接deep_input_emb和 dense_value_list
    output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(dnn_input)#经过若干层隐层变换，获得DNN的输出
    final_logit = tf.keras.layers.Dense(1, use_bias=False,
                                        kernel_initializer=tf.keras.initializers.glorot_normal(seed))(output)#获得DNN的logit

    output = PredictionLayer(task)(final_logit)#获得最终的预测结果

    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    return model
