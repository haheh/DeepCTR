'''
这个模块是用于构造特征列
feature_column.py中的类SparseFeat、DenseFeat、VarLenSparseFeat就是用来处理三种特征:类别特征、数值特征和序列特征。
我们只需要将原始特征转化为这三种特征列，之后就可以得到通用的特征输入，从而可调用models中的任意模型开始训练。
'''
import tensorflow as tf
from collections import namedtuple, OrderedDict# namedtuple:可命名的数组 
#OrderedDict:模块collections里面自带了一个子类OrderedDict，实现了对字典对象中元素的排序
from copy import copy
from itertools import chain# chain()可以把一组迭代对象串联起来

from tensorflow.python.keras.initializers import RandomNormal, Zeros
# keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None) 按照正态分布生成随机张量的初始化器。
#mean: 一个 Python 标量或者一个标量张量。要生成的随机值的平均数。
#stddev: 一个 Python 标量或者一个标量张量。要生成的随机值的标准差。
#seed: 一个 Python 整数。用于设置随机数种子。
# keras.initializers.Zeros() 将张量初始值设为 0 的初始化器。
from tensorflow.python.keras.layers import Input, Lambda
# Input():用于实例化 Keras 张量。英文文档 参数 8个，中文6个。返回值是一个张量
#以下参考中文文档：
#Keras 张量是底层后端(Theano, TensorFlow 或 CNTK) 的张量对象，我们增加了一些特性，使得能够通过知道模型的输入和输出来构建 Keras 模型。
# 例如，如果 a, b 和 c 都是 Keras 张量， 那么以下操作是可行的： model = Model(input=[a, b], output=c)
# 添加的 Keras 属性是： 
    # - _keras_shape: 通过 Keras端的尺寸推理 进行传播的整数尺寸元组。 
    # - _keras_history: 应用于张量的最后一层。 整个网络层计算图可以递归地从该层中检索。
#参数：
# shape: 一个尺寸元组（整数），不包含批量大小batch size（即几个）。 例如，shape=(32,) 表明期望的输入是按批次的 ‘32 维向量’。
# batch_shape: 一个尺寸元组（整数），包含批量大小。 例如，batch_shape=(10, 32) 表明期望的输入是 10 个 32 维向量。 
                                                 # batch_shape=(None, 32) 表明任意批次大小（即几个）的 32 维向量。
# name: 一个可选的层的名称的字符串。 在一个模型中应该是唯一的（不可以重用一个名字两次）。 如未提供，将自动生成。
# dtype: 输入所期望的数据类型，字符串表示 (float32, float64, int32...)
# sparse: 一个布尔值，指明需要创建的占位符是否是稀疏的。
# tensor: 可选的可封装到 Input 层的现有张量。 如果设定了，那么这个层将不会创建占位符张量。
# type_spec: A tf.TypeSpec object to create the input placeholder from. When provided, all other args except name must be None.
# ragged: A boolean specifying whether the placeholder to be created is ragged. Only one of 'ragged' and 'sparse' can be True. 
    # In this case, values of 'None' in the 'shape' argument represent ragged dimensions. 

# keras.layers.Lambda(function, output_shape=None, mask=None, arguments=None) 将任意表达式封装为 Layer 对象。
#function: 需要封装的函数。 将输入张量作为第一个参数。
#output_shape: 预期的函数输出尺寸。 只在使用 Theano 时有意义。 可以是元组或者函数。 
#arguments: 可选的需要传递给函数的关键字参数。
#输入尺寸：任意。当使用此层作为模型中的第一层时， 使用参数 input_shape （整数元组，不包括样本数的轴）。
#输出尺寸：由 output_shape 参数指定 (或者在使用 TensorFlow 时，自动推理得到)。
from .inputs import create_embedding_matrix, embedding_lookup, get_dense_input, varlen_embedding_lookup, \
    get_varlen_pooling_list, mergeDict
#varlen不定长
from .layers import Linear
#这个Linear是.layers.utils中定义的一个类，是tf.keras.layers.Layer类的继承
from .layers.utils import concat_func
#全连接层Fully Connection ：FC层在keras中叫做Dense层，在pytorch中叫Linear层。全连接的核心操作就是 矩阵向量乘积 。
#本质就是由一个特征空间线性变换到另一个特征空间。因此，dense层的目的是将前面提取的特征，在dense经过非线性变化，提取这些特征之间的关联，最后映射到输出空间上。
DEFAULT_GROUP_NAME = "default_group"

#处理 类别特征，稀疏
#SparseFeat用于处理类别特征，如性别、国籍等类别特征，将类别特征转为固定维度的稠密特征。
class SparseFeat(namedtuple('SparseFeat',
                            ['name', 'vocabulary_size', 'embedding_dim', 'use_hash', 'vocabulary_path', 'dtype', 'embeddings_initializer',
                             'embedding_name',
                             'group_name', 'trainable'])):
    '''
    处理类别特征，将其转为固定维度的稠密特征
    参数：
        name：生成的特征列的名字
        vocabulary_size：不同特征值的个数或当use_hash=True时的哈希空间
        embedding_dim：嵌入向量的维度
        use_hash：是否使用哈希编码，默认False
        dtype：默认int32
        embeddings_initializer：嵌入矩阵初始化方式，默认随机初始化
        embedding_name：默认None，其名字与name保持一致
        group_name：特征列所属的组
        traninable：嵌入矩阵是否可训练，默认True
    '''
    __slots__ = ()

    def __new__(cls, name, vocabulary_size, embedding_dim=4, use_hash=False, vocabulary_path=None, dtype="int32", embeddings_initializer=None,
                embedding_name=None,
                group_name=DEFAULT_GROUP_NAME, trainable=True):

        if embedding_dim == "auto":
            embedding_dim = 6 * int(pow(vocabulary_size, 0.25))
        if embeddings_initializer is None:
            embeddings_initializer = RandomNormal(mean=0.0, stddev=0.0001, seed=2020)

        if embedding_name is None:
            embedding_name = name

        return super(SparseFeat, cls).__new__(cls, name, vocabulary_size, embedding_dim, use_hash, vocabulary_path, dtype,
                                              embeddings_initializer,
                                              embedding_name, group_name, trainable)

    def __hash__(self):
        return self.name.__hash__()

#处理 可变长度序列 特征，eg文本特征
#处理类似文本序列的可变长度类型特征。
class VarLenSparseFeat(namedtuple('VarLenSparseFeat',
                                  ['sparsefeat', 'maxlen', 'combiner', 'length_name', 'weight_name', 'weight_norm'])):
    '''
    处理可变长度的SparseFeat，比如文本序列特征，对文本分词后，每个样本的文本序列包含的词数量是不统一的。
    参数：
        sparsefeat：属于SparseFeat的实例
        maxlen：所有样本中该特征列的长度最大值
        combiner：池化方法（mean,sum,max），默认是mean
        length_name：特征长度名字，如果是None的话，表示特征中的0是用来填充的
        weight_name：默认None，如果不为空，那么序列特征会与weight_name的权重特征进行相乘
        weight_norm：是否对权重分数做归一化，默认True

    '''
    __slots__ = ()

    def __new__(cls, sparsefeat, maxlen, combiner="mean", length_name=None, weight_name=None, weight_norm=True):
        return super(VarLenSparseFeat, cls).__new__(cls, sparsefeat, maxlen, combiner, length_name, weight_name,
                                                    weight_norm)
#因为传入的对象类型是SparseFeat，因此SparseFeat有的属性VarLenSparseFeat都有
    @property
    def name(self):
        return self.sparsefeat.name

    @property
    def vocabulary_size(self):
        return self.sparsefeat.vocabulary_size

    @property
    def embedding_dim(self):
        return self.sparsefeat.embedding_dim

    @property
    def use_hash(self):
        return self.sparsefeat.use_hash

    @property
    def vocabulary_path(self):
        return self.sparsefeat.vocabulary_path

    @property
    def dtype(self):
        return self.sparsefeat.dtype

    @property
    def embeddings_initializer(self):
        return self.sparsefeat.embeddings_initializer

    @property
    def embedding_name(self):
        return self.sparsefeat.embedding_name

    @property
    def group_name(self):
        return self.sparsefeat.group_name

    @property
    def trainable(self):
        return self.sparsefeat.trainable

    def __hash__(self):
        return self.name.__hash__()

#处理 数值特征，密集
#将稠密特征转为向量的形式，并使用transform_fn()对其做 归一化 操作或者其它的 线性或非线性 变换。
class DenseFeat(namedtuple('DenseFeat', ['name', 'dimension', 'dtype', 'transform_fn'])):
    """
    将稠密特征转为向量的形式，并使用transform_fn 函数对其做归一化操作或者其它的线性或非线性变换
    Args:
        name: 特征列名字
        dimension: 嵌入特征维度，默认是1
        dtype: 特征类型，default="float32"，
        transform_fn: 转换函数，可以是归一化函数，也可以是其它的线性变换函数，以张量作为输入，经函数处理后，返回张量
                        比如：lambda x: (x - 3.0) / 4.2)
    """
    """ Dense feature
    Args:
        name: feature name,
        dimension: dimension of the feature, default = 1.
        dtype: dtype of the feature, default="float32".
        transform_fn: If not `None` , a function that can be used to transform
        values of the feature.  the function takes the input Tensor as its
        argument, and returns the output Tensor.
        (e.g. lambda x: (x - 3.0) / 4.2).
    """
    __slots__ = ()

    def __new__(cls, name, dimension=1, dtype="float32", transform_fn=None):
        return super(DenseFeat, cls).__new__(cls, name, dimension, dtype, transform_fn)

    def __hash__(self):
        return self.name.__hash__()

    # def __eq__(self, other):
    #     if self.name == other.name:
    #         return True
    #     return False

    # def __repr__(self):
    #     return 'DenseFeat:'+self.name


def get_feature_names(feature_columns):
    '''
    作用：获取所有特征列的名字，以列表形式返回
    '''
    features = build_input_features(feature_columns)#为特征列构造keras tensor
    return list(features.keys())#返回特征列的names

def build_input_features(feature_columns, prefix=''):#prifix:前缀
    '''
    作用：为所有的特征列构造keras tensor，结果以OrderDict形式返回
    '''
    input_features = OrderedDict()
    for fc in feature_columns:
        #isinstance() 函数来判断一个对象是否是一个已知的类型，类似 type()。
        #不同于type()，isinstance() 会认为子类是一种父类类型，考虑继承关系。如果要判断两个类型是否相同 推荐使用isinstance()。
        if isinstance(fc, SparseFeat):#判断fc是否属于SparseFeat实例
            input_features[fc.name] = Input(
                shape=(1,), name=prefix + fc.name, dtype=fc.dtype)#Input()函数用于构造keras tensor
        elif isinstance(fc, DenseFeat):
            input_features[fc.name] = Input(
                shape=(fc.dimension,), name=prefix + fc.name, dtype=fc.dtype)
        elif isinstance(fc, VarLenSparseFeat):
            input_features[fc.name] = Input(shape=(fc.maxlen,), name=prefix + fc.name,
                                            dtype=fc.dtype)
            if fc.weight_name is not None:
                input_features[fc.weight_name] = Input(shape=(fc.maxlen, 1), name=prefix + fc.weight_name,
                                                       dtype="float32")
            if fc.length_name is not None:
                input_features[fc.length_name] = Input((1,), name=prefix + fc.length_name, dtype='int32')

        else:
            raise TypeError("Invalid feature column type,got", type(fc))

    return input_features


def get_linear_logit(features, feature_columns, units=1, use_bias=False, seed=1024, prefix='linear',
                     l2_reg=0, sparse_feat_refine_weight=None):
    '''
    作用：获取linear_logit（线性变换）的结果
    '''
    linear_feature_columns = copy(feature_columns)
    #将SparseFeat和VarLenSparseFeat的embedding_dim强制置换为1
    for i in range(len(linear_feature_columns)):
        if isinstance(linear_feature_columns[i], SparseFeat):
            linear_feature_columns[i] = linear_feature_columns[i]._replace(embedding_dim=1,
                                                                           embeddings_initializer=Zeros())
        if isinstance(linear_feature_columns[i], VarLenSparseFeat):
            linear_feature_columns[i] = linear_feature_columns[i]._replace(
                sparsefeat=linear_feature_columns[i].sparsefeat._replace(embedding_dim=1,
                                                                         embeddings_initializer=Zeros()))
    #获取用于线性变换的embedding list
    linear_emb_list = [input_from_feature_columns(features, linear_feature_columns, l2_reg, seed,
                                                  prefix=prefix + str(i))[0] for i in range(units)]
    #获取DenseFeat的数值特征
    _, dense_input_list = input_from_feature_columns(features, linear_feature_columns, l2_reg, seed, prefix=prefix)

    linear_logit_list = []
    for i in range(units):

        if len(linear_emb_list[i]) > 0 and len(dense_input_list) > 0:#既有稀疏类别特征也有稠密特征的情况
            sparse_input = concat_func(linear_emb_list[i])#将所有稀疏特征列的嵌入向量进行拼接
            dense_input = concat_func(dense_input_list)#将所有稠密特征列的数值特征进行拼接
            if sparse_feat_refine_weight is not None:
                sparse_input = Lambda(lambda x: x[0] * tf.expand_dims(x[1], axis=1))(
                    [sparse_input, sparse_feat_refine_weight])
            linear_logit = Linear(l2_reg, mode=2, use_bias=use_bias, seed=seed)([sparse_input, dense_input])#将sparse_input和dense_input拼接后进行线性变换
        elif len(linear_emb_list[i]) > 0:#仅有稀疏类别特征的情况
            sparse_input = concat_func(linear_emb_list[i])
            if sparse_feat_refine_weight is not None:
                sparse_input = Lambda(lambda x: x[0] * tf.expand_dims(x[1], axis=1))(
                    [sparse_input, sparse_feat_refine_weight])
            linear_logit = Linear(l2_reg, mode=0, use_bias=use_bias, seed=seed)(sparse_input)#此处应该是随机产生权重
        elif len(dense_input_list) > 0:#仅有稠密数值特征的情况
            dense_input = concat_func(dense_input_list)
            linear_logit = Linear(l2_reg, mode=1, use_bias=use_bias, seed=seed)(dense_input)
        else:   #empty feature_columns
            return Lambda(lambda x: tf.constant([[0.0]]))(list(features.values())[0])
        linear_logit_list.append(linear_logit)

    return concat_func(linear_logit_list)#将所有logit结果拼接后返回

#为所有特征列创建 嵌入矩阵 ，并分别返回包含SparseFeat和VarLenSparseFeat的 嵌入矩阵 的字典，和包含DenseFeat的 数值特征 的字典
def input_from_feature_columns(features, feature_columns, l2_reg, seed, prefix='', seq_mask_zero=True,
                               support_dense=True, support_group=False):
    '''
    为所有特征列创建嵌入矩阵，并分别返回包含SparseFeat和VarLenSparseFeat的嵌入矩阵的字典，和包含DenseFeat的数值特征的字典
    具体实现是通过调用inputs中的create_embedding_matrix、embedding_lookup、varlen_embedding_lookup等函数完成
    '''
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []

    embedding_matrix_dict = create_embedding_matrix(feature_columns, l2_reg, seed, prefix=prefix,
                                                    seq_mask_zero=seq_mask_zero)
    group_sparse_embedding_dict = embedding_lookup(embedding_matrix_dict, features, sparse_feature_columns)
    dense_value_list = get_dense_input(features, feature_columns)
    if not support_dense and len(dense_value_list) > 0:
        raise ValueError("DenseFeat is not supported in dnn_feature_columns")

    sequence_embed_dict = varlen_embedding_lookup(embedding_matrix_dict, features, varlen_sparse_feature_columns)
    group_varlen_sparse_embedding_dict = get_varlen_pooling_list(sequence_embed_dict, features,
                                                                 varlen_sparse_feature_columns)
    group_embedding_dict = mergeDict(group_sparse_embedding_dict, group_varlen_sparse_embedding_dict)
    if not support_group:
        group_embedding_dict = list(chain.from_iterable(group_embedding_dict.values()))
    return group_embedding_dict, dense_value_list
