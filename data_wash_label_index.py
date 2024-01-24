#等高线均值压缩
import os
import numpy as np
import pandas as pd
import time
from tqdm import tqdm  # 展示进度


def generate_new_var_data(dir_num,frame_internal,thred,area):

    # 初始化一个字典来存储每个变量对应每次降水的数据
    label_index=[]
    for dir_sort in tqdm(np.arange(dir_num),desc='data_dir Processing'):#这里改降雨次数！！！！！！
            #print(dgm_sort)
            # 指定文件夹路径
            folder_path = r'C:\Users\llush\Desktop\数模re\数模_原\数模data\NJU_CPOL_update2308\{}\{}.0km\data_dir_{}'.format('dBZ', "3",str(dir_sort).zfill(3))
            frames_num = len(os.listdir(folder_path))
            if frames_num>=20: #保证每次降水有20帧及以上
                for sample_index in range(frames_num - 2*frame_internal+1):
                    label=[]
                    #print("sample",sample_index)
                    for i in range(2*frame_internal):
                        if i>=frame_internal: #去除非强对流天气 这里的条件定义为后10帧最大值大于256（每帧满足强度大于thred的区域点数）
                            file_path= os.path.join(folder_path, "frame_{}.npy".format(str(sample_index + i).zfill(3)))  # 获取文件的完整路径
                            dBZ_npy_data = np.load(file_path)  # 使用NumPy加载.npy文件 [256,256]
                            label.append(np.count_nonzero(dBZ_npy_data > thred)) #label存储后10帧分别的区域点数
                    #print(var_sample_res.shape)
                    # print(len(label))
                    if np.max(label)>area:
                        label_index.append(1)
                        # print(np.array(var_sample).shape)
                    else:
                        label_index.append(0)

    label_index=np.array(label_index)
    # np.save('label_index.npy', label_index)
    print(np.sum(label_index))



generate_new_var_data(2,10,35,256)
