import tensorflow as tf
from PIL import Image
import os
import numpy as np

"""
此代码主要用于将图像数据转化为tfrecords文件
其中源图像的height*weight*channel=60*160*3
用于保存的图像格式为 height*weight*channel=60*160*1
图像的数据做过归一化的操作
---
usage:
当您使用的时候,只用修改flags中的三个参数即可;
如果您的源图像height*weight*channel不是60*160*3,请到get_image_label_batch() 中修改reshape()的参数
"""
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("pic_dir", "./source_data/train/", "源图片的路径")
tf.flags.DEFINE_string("letter", "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890", "验证码字符的种类")
tf.flags.DEFINE_string("tfrecords_dir", "./source_data/train.tfrecords", "tfrecords文件的保存路径及名字")

def dealWithLabel(labelStr_list):
    """

    :param label_list: ['2a2s', '2w3e', '4e5r', '3e5r', '4r5g'....]
    :return:
    """
    # 构建字符索引 {0：'a', 1:'b'......}
    num2letter = dict(enumerate(list(FLAGS.letter)))
    # 键值对翻转 {'a':0, 'b':1......}
    letter2num = dict(zip(num2letter.values(),num2letter.keys()))
    print(letter2num)

    # 构建标签的列表
    array = []
    for lablex in labelStr_list:

        letter_list = []

        for letter in list(str(lablex)): # '2a2s'>>[2,a,2,s]
            letter2_num = letter2num[str(letter)] #a>>2
            letter_list.append(letter2_num)

        array.append(np.asarray(letter_list).tostring())
    print(array)

    return array

def numpy_normalization(data):
    """
    用numpy实现对数组的归一化
    :param data: 数组
    :return:
    """

    range = np.max(data) - np.min(data)
    return (data-np.min(data))/range

def get_imagePath_label(path):
    """

    :param path: 图片的上级路径,如 './pic_source/train6000/'
    :return:
    image_path_list: ['./pic_source/train6000/2a2s.jpg', ''......]
    label_list: ["2a2s", ""......]
    """
    file_name_list = os.listdir(path=path) #结果>>>['2a2s.jpg', '', ''....]
    image_path_list = []
    labelStr_list = []
    for file_name in file_name_list:
        file_path = os.path.join(path,file_name)
        image_path_list.append(file_path)

        label_name = str(file_name).split(".")[0]
        labelStr_list.append(label_name)

    print("图像的目录:", image_path_list)
    print("标签列表:", labelStr_list)
    return image_path_list, labelStr_list

def get_image_label_batch(image_path_list, labelStr_list):
    """
    将图片集合,标签集合转换成最终的tensor集合
    :param image_path_list: ['../pic_source/train6000/2a2s.jpg', ' '......]
    :param labelStr_list: ["2a2s", "2s2d"......]
    :return:
    """

    image_batch = [] # 存放图片数据的数组
    # label_batch = [] # 存放图片标签的数组 注意存放的是代号 [2a2s]>>[3,24,3,56]
    for path in image_path_list:
        image = np.asarray(Image.open(path).convert('L')) #打开图片>>灰度化>>转为numpy数组
        image = numpy_normalization(image)    # 将数组归一化
        image2 = image.reshape((60,-1,1))  # 重塑形状
        image2 = image2.tostring()  # 转化为字符串
        image_batch.append(image2)

    label_batch = dealWithLabel(labelStr_list)  #numpy类型

    print('image_batch:', type(image_batch))
    print('label_batch:', type(label_batch))

    return image_batch, label_batch

def write2tfrecords(source_data_path = FLAGS.pic_dir, tfrecord_path=FLAGS.tfrecords_dir):
    """
     将图片内容和标签写入到tfrecords文件当中
    :param image_batch:特征值
    :param lable_batch:标签值
    :param 生成的tfrecord文件要存放的路径
    :return:
    """
    image_path_list, labelStr_list = get_imagePath_label(source_data_path)
    image_batch, lable_batch =get_image_label_batch(image_path_list, labelStr_list)


    # 建立TFRecords存储器
    writer = tf.python_io.TFRecordWriter(tfrecord_path)

    # 循环将每一个图片上的数据构造example协议块,序列化后写入
    for i in range(len(labelStr_list)):

        # 构造协议块
        example = tf.train.Example(features=tf.train.Features(feature={
            "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_batch[i]])),
            "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[lable_batch[i]]))
        }))

        writer.write(example.SerializeToString())
    # 关闭文件
    writer.close()
    print('生成TFRecord文件成功-----------------------------------------')

if __name__ == '__main__':
    # sess = tf.InteractiveSession()
    write2tfrecords()
    # a = np.array([[1,2,3],[12,2,34]]).tobytes()
    # a = np.frombuffer(a, dtype=np.int)
    # a = tf.constant(a)
    # print(a.eval())
