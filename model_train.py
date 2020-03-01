import numpy as np
import os
import tensorflow as tf
from cnn_utils import *
from data_util import *

tf.flags.DEFINE_integer("MAX_CAPTCHA", 4, "验证码共有4个字符")
tf.flags.DEFINE_integer("CHAR_SET_LEN", 62, "每个符号有62种选择,从A-Z,a-z,0-9")

def cnn(input,keep_prob=0.75,
                LABEL_NUM=4, LABEL_DEPTH=62,
                is_train=True
                ):
    """
    :param input: 输入图像的尺寸[-1,60,160,1]
    :param keep_prob: 每个元素被保留的概率
    :param MAX_CAPTCHA: 验证码共有4个字符
    :param CHAR_SET_LEN: 每个符号有62种选择,从A-Z,a-z,0-9
    :return:
    """
    # x = tf.reshape(tf.cast(input,tf.float32), shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    # todo 第一层卷积
    net = conv(name='conv1',input=input,w=[3, 3, 1, 32], b=32,is_train=is_train)  # 图像(-1,60,160,1) >>(60,160,32)
    net = max_pool(name='pool1',x=net,k=2) # 池化 >>(30,80,32)
    net = tf.nn.dropout(net, rate=1 - keep_prob) # dropout 防止过拟合

    # todo 第二层卷积
    net = conv(name='conv2', input=net,w=[3,3,32,64], b=64,is_train=is_train)  # (30,80,32) >>(30,80,64)
    net = max_pool(name='pool2',x=net,k=2) # 池化 >>[15,40,64]
    net = tf.nn.dropout(net, rate=1 - keep_prob) # dropout 防止过拟合

    # todo 第三层卷积
    net = conv(name='conv3', input=net,w=[3, 3, 64, 64], b=64,is_train=is_train) # (15,40,64) >>(15,40,64)
    net = max_pool(name='pool3', x=net,k=2) # 池化>>[8,20,64]
    net = tf.nn.dropout(net, rate=1 - keep_prob) # dropout 防止过拟合

    # todo Fully connected layer
    net = flatten(net)  # >>[-1, 8 * 20 * 64]
    net = tf.layers.dense(net, units=1024, activation=tf.nn.relu) # >>[-1,1024]
    net = tf.nn.dropout(net, rate=1 - keep_prob)

    net = tf.layers.dense(net, units=LABEL_NUM * LABEL_DEPTH, activation=None)  # >>[-1,4*62]
    return net

# 训练
def train_cnn(
                      trainData_dir='./source_data/train',  # 训练图像的目录
                      validData_dir= './source_data/valid',   # 测试图像的目录
                      model_save_path = './models/checkpoints',  # 模型保存路径
                      log_path='./models/logPath',         # 日志保存路径
                      label_num=4,    #标签共有四个值
                      label_depth=62, #每个标签位置有62中选择
                      image_height=60,
                      image_weight=160,
                      batch_size=50,
                      keep_prob=1.0,
                      lr=0.1,
                      epoches=60      #训练轮数
                     ):
    # 创建保存日志的文件夹
    create_dir_path(log_path)

    # 读取图片数据
    train_image, train_label = generateCnn_image_label_batch(image_path=trainData_dir, label_num=label_num, label_depth=label_depth,batch_size=batch_size,image_height=image_height,image_weight=image_weight)
    valid_image, valid_label = generateCnn_image_label_batch(image_path=validData_dir, label_num=label_num, label_depth=label_depth,batch_size=batch_size,image_height=image_height,image_weight=image_weight)

    training = tf.placeholder(tf.bool, shape=None, name='bn_training')  # 训练时需要更新参数,但测试时不需要更新参数
    train_or_valid = tf.placeholder_with_default(False, shape=None, name='is_train') # 判断是传入训练数据  还是 测试数据
    # 基于是否训练操作，做一个选择（选择训练数据集 或者 测试数据集）
    x = tf.cond(train_or_valid, lambda: train_image, lambda: valid_image)
    y = tf.cond(train_or_valid, lambda: train_label, lambda: valid_label)

    # 2、todo 调用cnn()函数构建模型---y_predict:[batch_size,4*62]
    y_predict = cnn(input=x, keep_prob=keep_prob,is_train=training)
    # 3,构建损失
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=y))
    # fixme 可视化模型损失
    tf.summary.scalar(name='train_loss', tensor=loss, collections=['train'])
    tf.summary.scalar(name='valid_loss', tensor=loss, collections=['valid'])
    # 4,构建优化器
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    # fixme 有批归一化都要这么用
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_opt = optimizer.minimize(loss)
    # 5,计算模型准确率
    equal_list = tf.equal(tf.argmax(tf.reshape(y_predict, [batch_size, label_num, label_depth]), 2), tf.argmax(tf.reshape(y, [batch_size, label_num, label_depth]), 2))
    accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))
    # fixme 可视化模型损失
    tf.summary.scalar(name='train_acc', tensor=accuracy, collections=['train'])
    tf.summary.scalar(name='valid_acc', tensor=accuracy, collections=['valid'])

    # 可视化代码
    train_summary = tf.summary.merge_all('train')
    val_summary = tf.summary.merge_all('valid')

    # 6、构建持久化路径 和 持久化对象。
    saver = tf.train.Saver(max_to_keep=2)
    create_dir_path(model_save_path)
    # print(y.name, train_opt.name, accuracy.name, loss.name)

    # 二、执行会话。
    with tf.Session() as sess:
        # 0、断点继续训练(恢复模型)
        ckpt = tf.train.get_checkpoint_state(model_save_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('加载持久化模型，断点继续训练!')
        else:
            # 1、初始化全局变量
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            print('没有持久化模型，从头开始训练!')

        # FileWriter 的构造函数中包含了参数log_dir，申明的所有事件都会写到它所指的目录下
        summary_writer = tf.summary.FileWriter(logdir=log_path, graph=sess.graph)


        # 开启协调器,使用start_queue_runners 启动队列填充
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)

        max_acc = 0  # 最高测试准确率测试
        step=1
        # 2、构建迭代的循环
        #****************************************************************************************************************************************
        try:
            for i in range(epoches):  # 迭代几轮
                for j in range(60):  # 共3000张图片,每个批次50张图片,共60批次
                    print('第{}轮,第{}个批次--------'.format(i+1,j+1))

                    _, train_loss, train_acc, train_summary_ = sess.run([train_opt, loss, accuracy, train_summary],feed_dict={training: True,     # 训练模式,参数可更新
                                                                                                                              train_or_valid:True # 训练模式,选择训练数据,而非测试数据
                                                                                                                              })
                    summary_writer.add_summary(train_summary_, global_step=step)
                    print(x.shape, y.shape)
                    # todo########################################################################
                    if (step) % 10 == 0: # 每10个批次用验证数据验证模型的精度,损失
                        _, valid_loss, valid_acc, valid_summary_ = sess.run([train_opt, loss, accuracy, val_summary],feed_dict={training: False,     # 验证模式,参数可更新
                                                                                                                                  train_or_valid:False # 验证模式,选择验证数据,而非测试数据
                                                                                                                                  })
                        summary_writer.add_summary(valid_summary_, global_step=step)
                        print('Epoch:{} - Step:{} - 验证数据损失:{:.5f} - 验证数据精度:{:.4f}'.format(i+1, j+1, valid_loss, valid_acc))
                        summary_writer.flush()
                    # todo########################################################################
                    if (step) % 5 == 0: # 每5个批次打印一次训练精度,损失; 测试精度,损失
                        print('Epoch:{} -Step:{} - 训练损失:{:.5f} - 训练精度:{:.4f}'.format(i+1, j+1, train_loss, train_acc))

                        # todo 模型持久化 每训练够5轮就保存一次模型
                        file_name = '_{}_model.ckpt'.format(step)
                        save_file = os.path.join(model_save_path, file_name)
                        saver.save(sess=sess, save_path=save_file, global_step=step)
                        print('model saved to path:{}'.format(save_file))

                    step += 1  # 每训练一批,step就+1,主要是为了tf.summary和模型持久化

        except tf.errors.OutOfRangeError:
            print("done")
        finally:
            coord.request_stop()
        coord.join(threads)
        # **************************************************************************************************************************************
        summary_writer.close()
    sess.close()
    print('训练运行成功.................................................')



def test_model(testData_dir='./source_data/test',  # 测试图像的目录
               model_save_path = './models/test_model/checkpoints', #假如现在开始训练,模型保存路径
               original_model_save_path = './models/checkpoints', #假如有之前训练好的模型,所在目录
               batch_size=10,
               label_num=4,
               label_depth=62,


                ):
    """
    调用持久化文件跑测试数据集的数据。（要求准确率在50%以上）
    """
    # 超参数:
    is_train=False #只预测,并计算精度,不更新参数
    keep_prob = 1.0
    lr = 0.001

    # 1,取出批次的图像及标签数据
    x,y=generateCnn_image_label_batch(image_path=testData_dir,batch_size=batch_size)
    # 2、构建cnn图（传入输入图片，获得logits）
    logits = cnn(input=x,keep_prob=keep_prob, is_train=is_train)
    # 3、构建损失函数
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=y))
    # 4、构建优化器。
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    # fixme 有批归一化都要这么用
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_opt = optimizer.minimize(loss)
    # 5、计算准确率
    equal_list = tf.equal(tf.argmax(tf.reshape(logits, [batch_size, label_num, label_depth]), 2), tf.argmax(tf.reshape(y, [batch_size, label_num, label_depth]), 2))
    accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))
    # 6、构建持久化模型的对象 并创建 持久化文件保存的路径
    saver = tf.train.Saver(max_to_keep=2)
    create_dir_path(model_save_path)


    # 二、构建会话
    with tf.Session() as sess:
        # 1、获取持久化的信息对象
        ckpt = tf.train.get_checkpoint_state(original_model_save_path)
        if ckpt is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)
            saver.recover_last_checkpoints(ckpt.all_model_checkpoint_paths)
            print('从持久化文件中恢复模型')
        else:
            print('没有持久化文件，从头开始训练!')

        # 开启协调器,使用start_queue_runners 启动队列填充
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)

        try:
            # 2、保存每个批次数据的准确率，再求平均值。
            test_acc_total = []
            # 3、构建迭代的循环
            for j in range(38):  # 总共有380张图片,每批次10张
                test_batch_acc = sess.run(accuracy)
                test_acc_total.append(test_batch_acc)
                if (j + 1) % 2 == 0:
                    print('第{}批次的测试准确率:{:.5f}'.format(j+1, np.mean(test_acc_total)))

        except tf.errors.OutOfRangeError:
            print("done")
        finally:
            coord.request_stop()
        coord.join(threads)

        # 随机打印一个例子

        sess.close()

if __name__ == '__main__':
    # train_cnn()
    test_model()
