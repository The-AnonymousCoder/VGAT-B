from Zero_watermarking import *
from NC import NC  # 使用用户提供的NC计算方法
import numpy as np
from matplotlib import pyplot as plt

'''Anti-scrambling watermark'''


def Arnold_Decrypt(image):
    shuffle_times, a, b = 10, 1, 1
    decode_image = np.zeros(shape=image.shape)
    h, w = image.shape[0], image.shape[1]
    N = h
    for time in range(shuffle_times):
        for ori_x in range(h):
            for ori_y in range(w):
                new_x = ((a * b + 1) * ori_x + (-b) * ori_y) % N
                new_y = ((-a) * ori_x + ori_y) % N
                decode_image[new_x, new_y] = image[ori_x, ori_y]
    return decode_image


def Construction(XLst, feature_num, List_Zero):
    List_Fea = len(List_Zero) * [0]
    # Construct feature matrix according to voting mechanism
    for i in range(0, feature_num):
        for j in range(i + 1, feature_num):
            Ni = len(XLst[i])
            Nj = len(XLst[j])
            index = (Ni * Nj) % len(List_Zero)
            ni = Ni % 2
            nj = Nj % 2
            if ni == nj:
                List_Fea[index] += 1
            else:
                List_Fea[index] += -1
    for p in range(0, len(List_Fea)):
        if List_Fea[p] > 0:
            List_Fea[p] = 255
        else:
            List_Fea[p] = 0
    return List_Fea


def XOR2(List_Fea, List_Zero):
    Lst_WaterMark = len(List_Zero) * [0]
    for m in range(0, len(List_Zero)):
        if List_Fea[m] == List_Zero[m]:
            Lst_WaterMark[m] = 0
        else:
            Lst_WaterMark[m] = 255
    return Lst_WaterMark


# NC值计算已移至NC.py文件中，使用标准归一化相关系数公式
# def NC(ori_img, decode_img):
#     h, w = ori_img.shape
#     S = 0
#     for i in range(0, h):
#         for j in range(0, w):
#             if ori_img[i][j] == decode_img[i][j]:
#                 S += 1
#             else:
#                 S += 0
#     nc = S / (h * w)
#     return nc


if __name__ == '__main__':
    img_0 = cv2.imread('Cat32.png', 0)
    img = cv2.imread("Zero_image.png", 0)  # 使用PNG保持二值性质
    img_deal = Watermark_deal(img)
    img_or = Watermark_deal(img_0)  #
    List_Zero = img_deal.flatten()
    fn_r = 'pso_data/BOUL.shp'
    XLst, YLst, feature_num = Read_Shapfile(fn_r)
    List_Fea = Construction(XLst, feature_num, List_Zero)
    Lst_WaterMark = XOR2(List_Fea, List_Zero)
    Re_mark = np.array(Lst_WaterMark).reshape(int(math.sqrt(len(List_Zero))), int(math.sqrt(len(List_Zero))))
    Decode_image = Arnold_Decrypt(Re_mark)  # Watermark after descrambling
    # Control descrambling times
    # for i in range(0, 9):
    #     Decode_image = Arnold_Decrypt(Decode_image)
    # 计算NC值 - 直接比较原始Cat32.png和提取的最终水印
    nc = NC(img_0, Decode_image)
    print(nc)
    # Show descrambled watermark
    plt.subplot(222)
    plt.imshow(Decode_image, 'gray')
    plt.title("Decode_image")
    
    plt.subplot(221)
    plt.imshow(img_0, 'gray')
    plt.title("Original")
    
    plt.savefig("watermark_extraction_results.png")
    plt.close()
    cv2.imwrite("Decode_image.png", Decode_image.astype(np.uint8))  # 使用PNG保持二值性质
    print(f"水印提取完成! NC值: {nc}")
    print(f"提取图像尺寸: {Decode_image.shape}")
