import cv2
import math
import numpy as np
from PIL import Image
# import matplotlib.pyplot as plt
# from os import listdir
import os
import random
# !unzip -oq /home/aistudio/dataset-dog10.zip

def DotMatrix(A, B):
    '''
    A,B:需要做乘法的两个矩阵，注意输入矩阵的维度ndim是否满足乘法要求（要做判断）
    '''
    A_col = A.shape[1]
    B_row = B.shape[0]

    dot_result = 0
    dot_matrix = np.zeros((A.shape[0], B.shape[1]))

    if A_col != B_row:
        return False

    else:
        for i in range(A.shape[0]):
            for j in range(B.shape[1]):

                for k in range(A.shape[1]):
                    temp = A[i][k] * B[k][j]
                    dot_result += temp

                dot_matrix[i][j] = dot_result
                dot_result = 0

    dot_matrix = np.array(dot_matrix)

    return dot_matrix


class Img:
    def __init__(self, image, rows, cols, center=[0, 0]):
        self.src = image  # 原始图像
        self.rows = rows  # 原始图像的行
        self.cols = cols  # 原始图像的列
        self.center = center  # 旋转中心，默认是[0,0]

    def Move(self, delta_x, delta_y):
        '''
        本函数处理生成做图像平移的矩阵
        '''
        self.transform = np.array([[1, 0, delta_x],
                                   [0, 1, delta_y],
                                   [0, 0, 1]])  # (TODO)

    def Zoom(self, factor):  # 缩放
        # factor>1表示缩小；factor<1表示放大
        self.transform = np.array([[factor, 0, 0],
                                   [0, factor, 0],
                                   [0, 0, 1]])  # (TODO)

    def Horizontal(self):
        '''水平镜像
        镜像的这两个函数，因为原始图像读进来后是height×width×3,和我们本身思路width×height×3相反
        所以造成了此处水平镜像和垂直镜像实现的效果是反的'''
        self.transform = np.array([[1, 0, 0], [0, -1, self.cols], [0, 0, 1]])  # (TODO)

    def Vertically(self):
        # 垂直镜像，注意实现原理的矩阵和最后实现效果是和水平镜像是反的
        self.transform = np.array([[-1, 0, self.rows], [0, 1, 0], [0, 0, 1]])
        # self.transform=np.array([[1,0,0],[0,1,rows],[0,0,1]])
        # self.transform=np.array([[-1,0,self.rows-1],[0,1,0],[0,0,1]])

    def Rotate(self, beta):  # 旋转
        # beta>0表示逆时针旋转；beta<0表示顺时针旋转
        self.transform = np.array([[math.cos(beta), math.sin(beta), 0],
                                   [-math.sin(beta), math.cos(beta), 0],
                                   [0, 0, 1]])

    def Process(self):

        # 初始化定义目标图像，具有3通道RBG值
        self.dst = np.zeros((self.rows, self.cols, 3), dtype=np.uint8)

        # 提供for循环，遍历图像中的每个像素点，然后使用矩阵乘法，找到变换后的坐标位置
        for i in range(self.rows):
            for j in range(self.cols):
                src_pos = np.array([[i - self.center[0]], [j - self.center[1]], [1]])  # 设置原始坐标点矩阵
                [[x], [y], [z]] = DotMatrix(self.transform, src_pos)  # 和对应变换做矩阵乘法

                x = int(x) + self.center[0]
                y = int(y) + self.center[1]

                if x >= self.rows or y >= self.cols or x < 0 or y < 0:
                    self.dst[i][j] = 255  # 处理未落在原图像中的点的情况
                else:
                    self.dst[i][j] = self.src[x][y]  # 使用变换后的位置


# base_path1 = '/home/aistudio/dog10'
# base_path2 = "C:/Users/ASUS/Desktop/dog10"
# base_path = "Z:\\archive\\Alzheimer_s Dataset"
#
# def get_pet_labels(image_dir):
#     # upath =
#     filtername_list = os.listdir(image_dir)
#
#     return filtername_list
#
# filtername_F = get_pet_labels(base_path)
# # print(filtername_F)
# filtername_dic = []
# for x in filtername_F:
#     if x != '.DS_Store' and x != '.ipynb_checkpoints' and x!= 'label_list.txt' and x!='test_list.txt' and x!='train_list.txt' and x!='validate_list.txt' and x!='1-Basset Hound' and x!='10-Shetlan' and x!='2-Beagle':
#         filtername_dic.append(base_path + '/' + x)
# print(filtername_dic)
# filtername_dic_end = []
# filtername_dic_temp = []
#
# for x in filtername_dic:
#     # print(x)
#     filtername_dic_temp = get_pet_labels(x)
#     # print(filtername_dic_temp)
#     for y in filtername_dic_temp:
#         # print(y)
#         tempppp = y.split('.')
#         # print(tempppp)
#         if y != '.DS_Store' and y != '.ipynb_checkpoints' and tempppp[1]=='jpeg':
#             filtername_dic_end.append(x + '/' + y)
#     filtername_dic_temp.clear()
# count=0
# for x in filtername_dic_end:
#     print(x)
# for x in filtername_dic_end:
#     if count == 222:
#         print(x)
#     if x=='C:/Users/ASUS/Desktop/dog10/3-Gray hound/dfaa17607c2568a4502f2b5657a8bae9.jpeg':
#         break
#     count+=1
# print(count)
# folders_name = listdir(base_path)
# print(filtername_dic)
# temp = filtername_dic_end[0]
# a = open(temp,"w")
count = 0

# for x in filtername_dic_end:
    # # x = filtername_dic_end[0]
    # # print(x)
    # tempp = x.split('\\')
    # # print(tempp)
    # # print(tempp)
    #
    # x = tempp[4]
    # print(x)

    # if(count==272):
        # print(x)
# x = "C:/Users/ASUS/Desktop/dog10/3-Gray hound/QQ20230313-130638.png" "Z:\archive\Alzheimer_s Dataset\test\MildDemented\26 (19).jpg"
# infer_path = r'' + x  # 要处理的单个图片地址
train_dir = "Z:\\archive (2)_\\Fire-Detection"
for i in os.listdir(train_dir):
    print(i)
    if i == '1':
        for j in os.listdir(train_dir+"/"+i):

            x = train_dir+"/"+i+"/"+j
            print(x)
            imgv = Image.open(x)# 打开图片
        # print(len(imgv.split()))

            imgv = imgv.convert("RGB")
            # plt.imshow(imgv) #根据数组绘制图像
            # plt.show() #显示图像

            rows = imgv.size[1]
            cols = imgv.size[0]
            # print(rows,cols)#注意此处rows和cols的取值方式

            imgv = np.array(imgv)  # 从图像生成数组

            img = Img(imgv, rows, cols, [0, 0])  # 生成一个自定Img类对象[0,0]代表处理的中心点

            img.Horizontal()  # 选择处理矩阵
            img.Process()  # 进行矩阵变换

            img2 = Image.fromarray(img.dst)  # 从处理后的数组生成图像
            temp = x.split('.')
            strs = temp[0] + '_hor' + '.' + temp[1]
            img2.save(strs)

            img.Vertically()  # 选择处理矩阵
            img.Process()  # 进行矩阵变换

            img2 = Image.fromarray(img.dst)  # 从处理后的数组生成图像
            temp = x.split('.')
            strs = temp[0] + '_ver' + '.' + temp[1]
            img2.save(strs)

            img.Move(-50, -50)  # 选择处理矩阵
            img.Process()  # 进行矩阵变换

            img2 = Image.fromarray(img.dst)  # 从处理后的数组生成图像
            temp = x.split('.')
            strs = temp[0] + '_move' + '.' + temp[1]
            img2.save(strs)

            img = Img(imgv, rows, cols, [int(rows / 2), int(cols / 2)])  # 生成一个自定Img类对象[0,0]代表处理的中心点

            img.Zoom(0.5)  # 选择处理矩阵
            img.Process()  # 进行矩阵变换

            img2 = Image.fromarray(img.dst)  # 从处理后的数组生成图像
            temp = x.split('.')
            strs = temp[0] + '_zoom' + '.' + temp[1]
            img2.save(strs)

            random_radian = random.randint(30, 180)
            img.Rotate(math.radians(random_radian))  # 旋转点选择图像大小的中心点
            img.Process()  # 进行矩阵变换

            img2 = Image.fromarray(img.dst)  # 从处理后的数组生成图像
            temp = x.split('.')
            strs = temp[0] + '_rotate' + '.' + temp[1]

            # strs = 'C:/Users/ASUS/Desktop/dog10/3-Gray hound/879cfa343b4c7514023a20dd7da4de1b_rotate.jpeg'
            img2.save(strs)
        # count+=1


#----------------------------------上面就是-------------------------------------------------------



# print(strs)
#     # plt.imshow(img2)
# plt.show()


# if __name__=='__main__':

#     infer_path=r'/home/aistudio/dog10/1-Basset Hound/5_5_N_5_巴吉度犬2.jpg'#要处理的单个图片地址
#     imgv = Image.open(infer_path)#打开图片
#     plt.imshow(imgv) #根据数组绘制图像
#     plt.show() #显示图像

#     rows = imgv.size[1]
#     cols = imgv.size[0]
#     print(rows,cols)#注意此处rows和cols的取值方式

#     imgv=np.array(imgv)#从图像生成数组

#     img=Img(imgv,rows,cols,[int(rows/2),int(cols/2)])#生成一个自定Img类对象[0,0]代表处理的中心点

#     img.Vertically() #选择处理矩阵
#     img.Process()#进行矩阵变换

#     img2=Image.fromarray(img.dst)#从处理后的数组生成图像
#     plt.imshow(img2)
#     plt.show()

#     # img.Horizontal() #镜像（0，0）
#     img.Move(-50,-50) #平移
#     img.Process()#进行矩阵变换


#     img2=Image.fromarray(img.dst)#从处理后的数组生成图像
#     plt.imshow(img2)
#     plt.show()

#     img.Horizontal() #镜像（0，0）
#     img.Process()#进行矩阵变换


#     img2=Image.fromarray(img.dst)#从处理后的数组生成图像
#     plt.imshow(img2)
#     plt.show()

#     random_radian = random.randint(30, 180)
#     img.Rotate(math.radians(random_radian))  #旋转点选择图像大小的中心点
#     img.Process()#进行矩阵变换


#     img2=Image.fromarray(img.dst)#从处理后的数组生成图像
#     plt.imshow(img2)
#     plt.show()

#     img.Zoom(0.5) #缩放
#     img.Process()#进行矩阵变换


#     img2=Image.fromarray(img.dst)#从处理后的数组生成图像
#     plt.imshow(img2)
#     plt.show()

# '''
# img.Vertically() #镜像(0,0)
# img.Horizontal() #镜像（0，0）
# img.Rotate(math.radians(180))  #旋转点选择图像大小的中心点
# img.Move(-50,-50) #平移
# img.Zoom(0.5) #缩放
# '''