# 用卷积神经网络训练验证码图片
### 单模型多标签
本项目使用CNN训练模型,算法架构接近于LeNet-5,预测标签如'2a2D'等.

使用CNN算法训练得到一个模型,一次性预测四个值.

项目目前存在的问题:只训练得到一个model,要预测四个值,训练的loss非常大,且收敛困难,基于本人的电脑GPU渣渣,大约2个小时的的迭代,但结果却不如人意.

如果您有强悍的算力资源,可否借我一用,不甚感激哈~

## 建议探讨:
建议使用CNN算法训练得到四个模型,分别预测四个值,以便于与降低训练难度, 加快模型的收敛速度, 模型精度或许也会更高.

后续我会基于上述想法用代码实现,欢迎关注哈~

如果您有更好的想法,欢迎骚扰,nanyangjx@126.com~


一,
  1,原始图像尺寸60 * 160 * 3,输入至模型的图像尺寸60*160*1,并且经过了归一化处理

  2,训练集(train)3000张图像,验证集(valid)1099张图像,测试集(test)380张图像

  3,图像的标签共有四个,如'2a2D','2Q3k'等,每个标签有62中选择(A-Z,a-z,0-9)

二,
  1,model文件夹主要存储模型,及运算日志.

  2,source_data文件夹主要存储图像数据

  3,tfread文件夹中的API主要是为了将图像转化为TFRead格式,但本框架并没用使用

  4,utils文件夹主要存储各类API

  5,gen_cap.py可直接运行,并生成验证码图片
  6,model_train.py进行模型的迭代运算

如果您有宝贵的意见,请及时联系我,我的邮箱:nanyangjx@126.com

if you have any question,please contact me directly,my e-mail adress:nanyangjx@126.com,thanks~
  
