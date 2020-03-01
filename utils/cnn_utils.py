import tensorflow as tf
import os

def create_dir_path(path):
    """
    创建文件夹
    :param path:
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print('成功创建路径:{}'.format(path))

def flatten(input_tensor):
    """
    flatten层，实现特征图 维度从 4-D  重塑到 2-D形状 [Batch_size, 列维度]
    :param input:
    :return:
    """
    shape = input_tensor.get_shape()
    flatten_shape = shape[1] * shape[2] * shape[3]
    flatten = tf.reshape(input_tensor, shape=[-1, flatten_shape])
    return flatten

def predict_to_onehot(label,label_num=4,label_depth=62):
    """
    将读取文件当中的目标值转换成one-hot编码
    :param label: [100, 4]      [[13, 25, 15, 15], [19, 23, 20, 16]......] >>[100, 4, 26]
    :param depth:每个验证码标签共有4个值,如"2a2s", 每个位置有62中选择
    :return: one-hot
    """
    # 进行one_hot编码转换，提供给交叉熵损失计算，准确率计算[100, 4, 62]
    # label = np.frombuffer(label, dtype=np.int)
    label_onehot = tf.one_hot(label, depth=label_depth, on_value=1.0, axis=-1)
    label_onehot = tf.reshape(label_onehot, [-1,label_num*label_depth])
    print(label_onehot)

    return label_onehot

def weight_variable(name, shape):
    # initial = tf.initializers.he_normal()
    initial = tf.truncated_normal(shape, mean=0.0, stddev=0.05, dtype=tf.float32)
    return tf.get_variable(name = name, shape = shape, initializer = initial)
    #return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def batch_norm(input,train_flag=True):
    return tf.contrib.layers.batch_norm(input, decay=0.9, center=True, scale=True, epsilon=1e-3,
                                        is_training=train_flag, updates_collections=None)

def conv(name,input,w,b, use_bn=True,is_train=True):
    # 卷积 + 批归一化+ Relu激活函数
    w_c = tf.Variable(tf.truncated_normal(w, stddev=0.1),name=name)
    b_c = tf.Variable(tf.truncated_normal([b], stddev=0.1), name=name)

    if use_bn:
        net = tf.nn.conv2d(input, w_c, strides=[1, 1, 1, 1], padding='SAME', name=name)
        net = tf.layers.batch_normalization(net, training=is_train)
    else:
        net = tf.nn.conv2d(input, w_c, strides=[1, 1, 1, 1], padding='SAME', name=name)

    net = tf.nn.relu(net)

    return net

def max_pool(name,x,k):
    return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME',name=name)



