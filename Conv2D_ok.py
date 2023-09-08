import numpy as np


def activation_zou(str, data, alpha=0.01):
    if str == 'relu':
        return np.maximum(0, data)
    elif str == 'leaky_relu':
        return np.where(data > 0, data, alpha * data)
    elif str == 'tanh':
        return np.tanh(data)
    elif str == 'sigmoid':
        return np.exp(data) / (1 + np.exp(data))


def Donv2D_yi(input, flatten, kernel_shape, activtion):
    # 创建一个3D输入数据，假设它是一个4x4x4的立方体矩阵
    input_volume = input
    input_shape = input_volume.shape

    flatten_number = flatten

    # 创建一个3D卷积核，假设它是一个2x2x2的立方体矩阵
    kernel_line = kernel_shape[0]
    kernel_row = kernel_shape[1]

    if input_volume.ndim == 3:
        kernel = np.random.uniform(-1, 1, (flatten_number, input_shape[0], kernel_line, kernel_row))
        # 获取输入数据和卷积核的形状
        kernel_shape = kernel.shape

        # 计算卷积后的立方体的形状
        output_shape = (
            flatten_number,
            input_shape[0] - kernel_shape[1] + 1,
            input_shape[1] - kernel_shape[2] + 1,
            input_shape[2] - kernel_shape[3] + 1
        )

        # 创建用于存储卷积结果的立方体
        output_volume = np.zeros(output_shape)

        # 执行三维卷积操作
        for z in range(flatten_number):
            for i in range(output_shape[1]):
                for j in range(output_shape[2]):
                    for k in range(output_shape[3]):
                        sub_volume = input_volume[i:i + kernel_shape[1], j:j + kernel_shape[2], k:k + kernel_shape[3]]
                        output_volume[z, i, j, k] = np.sum(sub_volume * kernel[z])

        print("卷积结果的形状:", output_shape)
        print("卷积结果:")
        print(output_volume)

        output_volume = activation_zou(activtion, output_volume)
        print("池化效果：")
        print(output_volume)

        return output_volume

    elif input_volume.ndim == 2:
        kernel = np.random.uniform(-1, 1, (flatten_number, kernel_line, kernel_row))
        # print(kernel)

        # 获取输入图像和卷积核的尺寸
        image_height, image_width = input_volume.shape
        flatten_number, kernel_height, kernel_width = kernel.shape

        # 计算卷积后的图像尺寸
        output_height = image_height - kernel_height + 1
        output_width = image_width - kernel_width + 1

        # 创建用于存储卷积结果的矩阵
        output_volume = np.zeros((flatten_number, output_height, output_width))

        # 执行卷积操作
        for z in range(flatten_number):
            for i in range(output_height):
                for j in range(output_width):
                    patch = input_volume[i:i + kernel_height, j:j + kernel_width]
                    output_volume[z, i, j] = np.sum(patch * kernel[z])

        print("卷积结果:")
        print(output_volume)

        output_volume = activation_zou(activtion, output_volume)
        print("池化效果：")
        print(output_volume)

        return output_volume

    else:
        # "\x1b[31mWarning!!!!请在二维数据和三维数据间选择！！！\x1b[0m"
        print("\x1b[31mWarning!!!!请在二维数据和三维数据间选择！！！\x1b[0m")
        return -1

image = np.random.randint(0, 1, (3, 28, 28))
print(image.ndim)

output = Donv2D_yi(image, 32, (2, 2), activtion='relu')
print(output.shape)
