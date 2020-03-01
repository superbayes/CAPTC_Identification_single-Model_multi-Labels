
import tensorflow as tf
import numpy as np
import os

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("captcha_dir", "./tfrecords/captcha.tfrecords", "验证码数据的路径")

def read_and_decode(tfrecords_dir,
                    batch_size=50,  # 训练集每个批次的样本个数
                    train_whole_sample_size=3000  # 训练集总量
                    ):
    """
    读取验证码数据API
    :return: image_batch, label_batch
    """
    # 1、构建文件队列
    file_queue = tf.train.string_input_producer([tfrecords_dir])

    # 2、构建阅读器，读取文件内容，默认一个样本
    reader = tf.TFRecordReader()

    # 读取内容
    key, value = reader.read(file_queue)

    # tfrecords格式example,需要解析
    features = tf.parse_single_example(value, features={
        "image": tf.FixedLenFeature([], tf.string),
        "label": tf.FixedLenFeature([], tf.string),
    })

    # 解码内容，字符串内容
    # 1、先解析图片的特征值
    image = tf.decode_raw(features["image"], tf.float32)
    # 1、先解析图片的目标值
    label = tf.decode_raw(features["label"], tf.uint8)

    # print(image, label)

    # 改变形状
    image_reshape = tf.reshape(image, [60, 160, 1])
    label_reshape = tf.reshape(label, [4])

    # 进行批处理,每批次读取的样本数 100, 也就是每次训练时候的样本
    # image_batch, label_btach = tf.train.batch([image_reshape, label_reshape], batch_size=batch_size, num_threads=1, capacity=batch_size)
    # 7、将其转换为队列的批次获取操作
    image, label = tf.train.shuffle_batch(
        [image_reshape, label_reshape],
        batch_size=batch_size,
        num_threads=2,
        capacity=train_whole_sample_size,
        min_after_dequeue=None
    )
    # print(image_reshape, label_reshape)
    # return image_batch, label_btach
    print(image, label)
    return image, label



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

def create_dir_path(path):
    """
    创建文件夹
    :param path:
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print('成功创建路径:{}'.format(path))


