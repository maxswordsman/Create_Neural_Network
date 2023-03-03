"""
    将自己手写的数字(PNG存储在my_image中)进行处理并且进行数据存储(存储到training_data_list中)
        1.书中的源码:https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork/blob/master/part3_load_own_images.ipynb
        2.将自己创建的数字图片存储在 my_image，每张图片对应的label为图片的上层文件夹，图片格式为xxx.PNG
        文件结构;
         my_image
            ├── 0
                └── 0.png
            ├── 1
                └── 1.png
            ├── 2
                └── 2.png
            ├── 3
                └── 3.png
            ├── 4
                └── 4.png
            ├── 5
                └── 5.png
            ├── 6
                └── 6.png
            ├── 7
                └── 7.png
            ├── 8
                └── 8.png
            └── 9
                └── 9.png
"""

import numpy
import matplotlib.pyplot as plt
import PIL
import os
import torchvision




def my_image(file_path):
    # 将图片放缩至 （28，28）
    ReSize = torchvision.transforms.Resize((28, 28))
    # 用于存储数据的列表
    training_data_list = []
    # 将路径下的文件转化 列表 --- 图片的标签列表
    num_class = [cla for cla in os.listdir(file_path)]
    for cla in num_class:
        # 每一个标签文件夹
        cla_path = os.path.join(file_path,cla)
        # 每一个标签文件夹下的图片列表
        images = os.listdir(cla_path)
        for image in images:
            # 每一张图片的路径
            image_path = os.path.join(cla_path,image)

            # 开始对数据进行处理
            # 将图片转为灰度图片
            image_array = PIL.Image.open(image_path).convert('L')
            # 将图片的大小缩放到 （28，28）
            image_array_crop = ReSize(image_array)
            # 将 PIL.Image.Image 数据类型变为 ndarry 二维数组类型 ---- 并将其展平为一维数组(784)列 ---网络中输入数据的固定格式
            image_data = numpy.asfarray(image_array_crop).reshape(784)
            # 像素值0表示黑色  255表示白色 但是 MNIST 中相反因此需要用 255-image_data
            image_data = 255 - image_data
            # 数据归一化并且进行偏移0.01，防止0输入造成梯度消失
            image_data = (image_data/255.0 * 0.99) + 0.01
            record = numpy.append(float(cla),image_data)
            training_data_list.append(record)

    return training_data_list












# 与 Network.py中的 train_data_list 效果一致
training_data_list = my_image("/home/zxz/Proj/deeplearning/Create_neural_network/my_image")
print(training_data_list[0])

"""
# 可视化图片
    plt.imshow(image_data.reshape(28,28), cmap='Greys', interpolation='None')
    plt.show()
"""









