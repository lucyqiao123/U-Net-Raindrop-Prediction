#等高线均值压缩
import os
import numpy as np
import time
from tqdm import tqdm  # 展示进度


def generate_new_var_data(dir_num,frame_internal,thred,area):
    norm_param = {'dBZ': [0, 65],
                  'ZDR': [-1, 5],
                  'KDP': [-1, 6]}
    # dgm = np.array(['3'])
    # 初始化一个字典来存储每个变量对应每次降水的数据
    var_data = {}
    label_index=np.load("label_index.npy")
    # print(np.sum(label_index))
    for var_sort in norm_param.keys():

        var_data[var_sort] = {}  # 初始化对应的字典
        var_sample=[]
        if_index = 0
        sample_num=0
        for dir_sort in tqdm(np.arange(dir_num),desc='data_dir Processing'):#降雨次数
            # for dgm_sort in dgm:
                #print(dgm_sort)
                # 指定文件夹路径
            folder_path = r'C:\Users\llush\Desktop\数模re\数模_原\数模data\NJU_CPOL_update2308\{}\{}.0km\data_dir_{}'.format(var_sort, "3",str(dir_sort).zfill(3))
            frames_num = len(os.listdir(folder_path))

            if frames_num>=20: #保证每次降水有20帧及以上
                for sample_index in range(frames_num - 2*frame_internal+1):
                    # print(var_sort,if_index)
                    label=[]
                    #print("sample",sample_index)
                    var_sample_res = []
                    if label_index[if_index] == 1:
                        sample_num+=1
                        print(var_sort, sample_num)
                        for i in range(2*frame_internal):
                            # print(i)
                            file_path = os.path.join(folder_path, "frame_{}.npy".format(str(sample_index + i).zfill(3)))  # 获取文件的完整路径
                            var_sample_res.append(np.load(file_path))  # 最后一次循环得到[20,256,256]
                        var_sample.append(var_sample_res) #[总样本组数,20,256,256]

                        if sample_num%200==0 or sample_num==np.sum(label_index):#边写边存储，删除变量，减少内存占用
                            mmin, mmax = norm_param[var_sort]
                            var_sample =(np.array(var_sample)-mmin) / (mmax - mmin)
                            np.save(r'C:\Users\llush\Desktop\数模re\数模_原\模型实现代码附件\data_after_wash_new\{}_sample{}.npy'.format(var_sort,sample_num),var_sample) #样本index从1开始的
                            del var_sample #释放内存
                            var_sample=[]
                    if_index += 1
                        # print(np.array(var_sample).shape)
    print("--------------数据加载完成----------------")
    # print(np.load("dBZ_sample5.npy").shape)
    # print(np.load("ZDR_sample5.npy").shape)

generate_new_var_data(258,10,35,256)
