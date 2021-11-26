# -*- coding:utf-8 -*-
"""

Author:
    Weichen Shen,weichenswc@163.com

"""
'''
SparseFeat和VarLenSparseFeat对象需要创建嵌入矩阵，嵌入矩阵的构造和查表等操作都是通过inputs.py模块实现的
处理输入，构造嵌入矩阵并实现了查表功能
'''


from collections import defaultdict
#Python中通过Key访问字典，当Key不存在时，会引发‘KeyError’异常。为了避免这种情况，使用collections类中的defaultdict()方法来为字典提供默认值。
from itertools import chain

from tensorflow.python.keras.layers import Embedding, Lambda
#keras.layers.Embedding(input_dim, output_dim, embeddings_initializer='uniform', embeddings_regularizer=None, 
    # activity_regularizer=None, embeddings_constraint=None, mask_zero=False, input_length=None)
#Embedding层接收的输入是一个二维矩阵(samples, sequence_length)，即样本集的作分词处理后的单词的标记化的表示，其中 samples是样本总数，
    # sequence_length是标记总数，矩阵中每一个元素代表一个标记，Embedding层接收输入并为每一个元素匹配一个数值向量，
    # 最后输出一个三维向量(samples, sequence_length, output_dim)。
#将正整数（索引值）转换为固定尺寸的稠密向量。 例如： [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]。该层只能用作模型中的第一层。
#----input_dim：在分词过程中每一个标记都有唯一一个区别于其它标记的索引，input_dim = 所有标记的索引的最大值 + 1
#----output_dim: 数值向量的维度
#input_dim: int > 0。字典大小， 即，最大整数 index + 1。torch.nn.Embedding()中：词典的大小尺寸。比如总共出现5000个词，那就输入5000。此时index为（0-4999）。
#output_dim: int >= 0。词向量的维度。torch.nn.Embedding()中：嵌入向量的维度，即用多少维来表示一个符号。
#embeddings_initializer: embeddings 矩阵的初始化方法。
#embeddings_regularizer: embeddings matrix 的正则化方法。
#embeddings_constraint: embeddings matrix 的约束函数。
#mask_zero: 是否把 0 看作为一个应该被遮蔽的特殊的 "padding" 值。 这对于可变长的 循环神经网络层 十分有用。在计算中, 是否屏蔽这些填补0值的作用。
    # 如果设定为 True，那么接下来的所有层都必须支持 masking，否则就会抛出异常。 
    # 如果 mask_zero 为 True，作为结果，索引 0 就不能被用于字典中 （input_dim 应该与 vocabulary + 1 大小相同）。
#input_length: 输入序列的长度，当它是固定的时。 如果你需要连接 Flatten 和 Dense 层，则这个参数是必须的 （没有它，dense 层的输出尺寸就无法计算）。
#输入尺寸:尺寸为 (batch_size, sequence_length) 的 2D 张量，即矩阵。
#输出尺寸:尺寸为 (batch_size, sequence_length, output_dim) 的 3D 张量。
#一个张量就是一个容器，可以存储 N 维的数据及其线性运算。经常有人错误地将张量和矩阵交替使用（确切地说，矩阵是一个2维张量），张量是 N 维空间的矩阵的泛指。
#PyTorch中的Tensor类，它相当于Numpy中的ndarray
from tensorflow.python.keras.regularizers import l2

from .layers.sequence import SequencePoolingLayer, WeightedSequenceLayer
#SequencePoolingLayer用于对变长序列特征或多值特征进行池化操作（相加，平均，最大）
#WeightedSequenceLayer用于对变长序列特征或多值特征进行权重分数（weight score）
from .layers.utils import Hash
#继承tf.keras.layers.Layer
# 在设置“vocabulary_path”时查找table中的keys，并输出相应的值。
# 如果未设置“vocabulary_path”，则“Hash”将把输入哈希到[0，num_bucket）。
# 当“mask_zero”=True时，输入值“0”或“0.0”将设置为“0”，其他值将在范围[1，num_bucket]中设置。
# 下面的代码段使用“vocabulary_path”文件初始化“Hash”，第一列作为键，第二列作为值：

    # * `1,emerson`
    # * `2,lake`
    # * `3,palmer`

    # >>> hash = Hash(
    # ...   num_buckets=3+1,
    # ...   vocabulary_path=filename,
    # ...   default_value=0)
    # >>> hash(tf.constant('lake')).numpy()
    # 2
    # >>> hash(tf.constant('lakeemerson')).numpy()
    # 0

# Args：
# num_buckets:一个>=1的整数。当设置了“vocabulary_path”，buckets的数量或vocabulary的大小+1。
# mask_zero:默认值为False。当'mask_zero'为'True'，“Hash”值将在以下情况下将输入“0”或“0.0”哈希为值“0”。
    # 在设置了'vocabulary_path'时'mask_zero'不被使用。
# vocabulary_path：默认为“无”。字典哈希的“CSV”文本文件路径，包含两列，由分隔符“逗号”分隔，第一列是值，第二列是键。
    # 键数据类型为'string'，值数据类型为'int'。路径必须可以从初始化了“Hash”的任何位置访问。
# default_value：默认值“0”。table中缺少键时的默认值。
# **kwargs：附加关键字参数。


def get_inputs_list(inputs):
    '''
    作用：过滤输入中的空值并返回列表形式的输入
    filter()过滤输入中的空值
    map()是取每个元素x的value
    chain()构建了一个迭代器，循环处理输入中的每条样本
    最后返回一个list
    '''
    return list(chain(*list(map(lambda x: x.values(), filter(lambda x: x is not None, inputs)))))
# >>> foo = [2, 18, 9, 22, 17, 24, 8, 12, 27]
# >>>
# >>> print filter(lambda x: x % 3 == 0, foo)
# [18, 9, 24, 12, 27]
# >>>
# >>> print map(lambda x: x * 2 + 10, foo)
# [14, 46, 28, 54, 44, 58, 26, 34, 64]

def create_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns, seed, l2_reg,
                          prefix='sparse_', seq_mask_zero=True):
    '''
    作用：为每个稀疏特征创建可训练的嵌入矩阵，使用字典存储所有特征列的嵌入矩阵，并返回该字典
    '''
    sparse_embedding = {}
    #处理稀疏特征
    for feat in sparse_feature_columns:
        # 为每个稀疏特征初始化一个vocabulary_size * embedding_dim 大小的嵌入矩阵
        emb = Embedding(feat.vocabulary_size, feat.embedding_dim,
                        embeddings_initializer=feat.embeddings_initializer,
                        embeddings_regularizer=l2(l2_reg),
                        name=prefix + '_emb_' + feat.embedding_name)
        # 令该嵌入矩阵可训练
        emb.trainable = feat.trainable
        #添加到字典中
        sparse_embedding[feat.embedding_name] = emb
    #处理可变长度稀疏特征，处理方法同上
    if varlen_sparse_feature_columns and len(varlen_sparse_feature_columns) > 0:
        for feat in varlen_sparse_feature_columns:
            # if feat.name not in sparse_embedding:
            emb = Embedding(feat.vocabulary_size, feat.embedding_dim,
                            embeddings_initializer=feat.embeddings_initializer,
                            embeddings_regularizer=l2(
                                l2_reg),
                            name=prefix + '_seq_emb_' + feat.name,
                            mask_zero=seq_mask_zero)
            emb.trainable = feat.trainable
            sparse_embedding[feat.embedding_name] = emb
    return sparse_embedding


def get_embedding_vec_list(embedding_dict, input_dict, sparse_feature_columns, return_feat_list=(), mask_feat_list=()):
    '''
    作用：从所有稀疏特征列中查询指定稀疏特征列(参数return_feat_list）的嵌入矩阵，以列表形式返回查询结果
    关键参数：
        embedding_dict：type：dict；存储着所有特征列的嵌入矩阵的字典
        input_dict：type：dict；存储着特征列和对应的嵌入矩阵索引的字典，在没有使用hash查询时使用 
        sparse_feature_columns：type：list；所有稀疏特征列
        return_feat_list:需要查询的特征列，默认为空，为空则返回所有稀疏特征列的嵌入矩阵，不为空则仅返回该元组中的特征列的嵌入矩阵
    '''
    embedding_vec_list = []
    for fg in sparse_feature_columns:
        feat_name = fg.name
        if len(return_feat_list) == 0 or feat_name in return_feat_list:
            if fg.use_hash:
                lookup_idx = Hash(fg.vocabulary_size, mask_zero=(feat_name in mask_feat_list), vocabulary_path=fg.vocabulary_path)(input_dict[feat_name])
            else:
                lookup_idx = input_dict[feat_name]

            embedding_vec_list.append(embedding_dict[feat_name](lookup_idx))

    return embedding_vec_list


def create_embedding_matrix(feature_columns, l2_reg, seed, prefix="", seq_mask_zero=True):
    '''
    作用：从所有特征列中筛选出SparseFeat和VarLenSparseFeat，然后调用函数create_embedding_dict为筛选的特征列创建嵌入矩阵
    '''
    from . import feature_column as fc_lib

    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, fc_lib.SparseFeat), feature_columns)) if feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, fc_lib.VarLenSparseFeat), feature_columns)) if feature_columns else []
    sparse_emb_dict = create_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns, seed,
                                            l2_reg, prefix=prefix + 'sparse', seq_mask_zero=seq_mask_zero)
    return sparse_emb_dict


def embedding_lookup(sparse_embedding_dict, sparse_input_dict, sparse_feature_columns, return_feat_list=(),
                     mask_feat_list=(), to_list=False):
    '''
    作用：从所有稀疏特征列中查询指定稀疏特征列(参数return_feat_list）的嵌入矩阵，以字典形式返回查询结果
    参数：
        sparse_embedding_dict：存储稀疏特征列的嵌入矩阵的字典
        sparse_input_dict：存储稀疏特征列的名字和索引的字典
        sparse_feature_columns：稀疏特征列列表，元素为SparseFeat
        return_feat_list：需要查询的稀疏特征列，如果元组为空，默认返回所有特征列的嵌入矩阵
        mask_feat_list：用于哈希查询
        to_list：是否以列表形式返回查询结果，默认是False
    '''
    group_embedding_dict = defaultdict(list)#存储结果的列表
    for fc in sparse_feature_columns:# 遍历查找
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if (len(return_feat_list) == 0 or feature_name in return_feat_list):
            if fc.use_hash:#获取哈希查询的索引
                lookup_idx = Hash(fc.vocabulary_size, mask_zero=(feature_name in mask_feat_list), vocabulary_path=fc.vocabulary_path)(
                    sparse_input_dict[feature_name])
            else:#从sparse_input_dict中获取该特征列的索引
                lookup_idx = sparse_input_dict[feature_name]

            group_embedding_dict[fc.group_name].append(sparse_embedding_dict[embedding_name](lookup_idx))
    if to_list:#如果为真，则将结果转为列表形式返回
        return list(chain.from_iterable(group_embedding_dict.values()))
    return group_embedding_dict


def varlen_embedding_lookup(embedding_dict, sequence_input_dict, varlen_sparse_feature_columns):
    '''
    作用：获取varlen_sparse_feature_columns的嵌入矩阵
    '''
    varlen_embedding_vec_dict = {}
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if fc.use_hash:
            lookup_idx = Hash(fc.vocabulary_size, mask_zero=True, vocabulary_path=fc.vocabulary_path)(sequence_input_dict[feature_name])
        else:
            lookup_idx = sequence_input_dict[feature_name]
        varlen_embedding_vec_dict[feature_name] = embedding_dict[embedding_name](lookup_idx)
    return varlen_embedding_vec_dict


def get_varlen_pooling_list(embedding_dict, features, varlen_sparse_feature_columns, to_list=False):
    '''
    作用：获取varlen_sparse_feature_columns池化后的嵌入向量
    '''
    # 常见的池化有 最大池化(Max Pooling) , 平均池化(Average Pooling) ,使用池化函数来进一步对卷积操作得到的特征映射结果进行处理。
    # 池化会将平面内某未知及其相邻位置的特征值进行统计汇总。并将汇总后的结果作为这一位置在该平面的值。最大池化会计算该位置及其相邻矩阵区域内的最大值，
    # 并将这个最大值作为该位置的值，平均池化会计算该位置及其相邻矩阵区域内的平均值，并将这个值作为该位置的值。使用池化不会造成数据矩阵深度的改变，
    # 只会在高度和宽带上降低，达到降维的目的。
    pooling_vec_list = defaultdict(list)
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        combiner = fc.combiner
        feature_length_name = fc.length_name
        if feature_length_name is not None:# length_name不为空，说明该特征列不存在用0填充的情况
            if fc.weight_name is not None:# weight_name不为空，说明序列需要进行权重化操作
                seq_input = WeightedSequenceLayer(weight_normalization=fc.weight_norm)(
                    [embedding_dict[feature_name], features[feature_length_name], features[fc.weight_name]])#需要对查找结果做权重化操作再得到seq_input
            else:# weight_name为空，说明序列不需要进行权重化操作
                seq_input = embedding_dict[feature_name]#直接从嵌入矩阵里找到对应结果,赋值给seq_input
            vec = SequencePoolingLayer(combiner, supports_masking=False)(
                [seq_input, features[feature_length_name]])#池化操作，因为没有填充，所以supports_masking=False，即池化时不需要mask掉填充的部分
        else:#length_name为空，说明该特征列存在用0填充的情况，因此在权重化操作和池化操作时都要令supports_masking=True，即mask掉填充的部分
            if fc.weight_name is not None:
                seq_input = WeightedSequenceLayer(weight_normalization=fc.weight_norm, supports_masking=True)(
                    [embedding_dict[feature_name], features[fc.weight_name]])
            else:
                seq_input = embedding_dict[feature_name]
            vec = SequencePoolingLayer(combiner, supports_masking=True)(
                seq_input)
        pooling_vec_list[fc.group_name].append(vec)
    if to_list:
        return chain.from_iterable(pooling_vec_list.values())
    return pooling_vec_list


def get_dense_input(features, feature_columns):
    '''
    作用：从所有特征列中选出DenseFeat，并以列表形式返回结果
    '''
    from . import feature_column as fc_lib
    #筛选出DenseFeat元素
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, fc_lib.DenseFeat), feature_columns)) if feature_columns else []
    dense_input_list = []
    #循环对各个DenseFeat元素执行transform_fn()
    for fc in dense_feature_columns:
        if fc.transform_fn is None:
            dense_input_list.append(features[fc.name])
        else:
            transform_result = Lambda(fc.transform_fn)(features[fc.name])
            dense_input_list.append(transform_result)
    return dense_input_list


def mergeDict(a, b):
    '''
    作用：将a、b两个字典合并
    '''
    c = defaultdict(list)
    for k, v in a.items():
        c[k].extend(v)
    for k, v in b.items():
        c[k].extend(v)
    return c
