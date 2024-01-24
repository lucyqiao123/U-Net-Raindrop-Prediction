# U-Net Raindrop Prediction
复刻论文：Improving Nowcasting of Convective Development by Incorporating Polarimetric Radar Variables Into a Deep‐Learning Model.Geophysical Research Letters,48(21), e2021GL095302

数据：论文supporting information/2023年华为杯中国研究生数学建模竞赛F题官网公告

技术文件路径：数据清洗，筛选合适样本data_wash_label_index.py--->产出样本标签label_index.npy-->存储样本数据var_data_filter_re.py-->模型U_net.py

预测结果：结合loss和CSI指标,最优epoch为2;train和test最终loss收敛于0.01左右,CSI最大为train样本中一帧为0.62，基本符合论文结果

目前存在问题：复刻论文提到使用B-MSE，作者内存不够没用上；train、test样本分布不均衡，训练效果由样本均衡性决定，换一批样本test效果就有所变化；存在过拟合问题，还需调参；
本来应该用全部train和test样本跑，但时间不够，就各自随机选取了200个样本，效果好像也还行

*仅迭代了10个epcoh，大致看走势

*前后10帧降雨可视化
![image](https://github.com/lucyqiao123/U-Net-Raindrop-Prediction/blob/main/visdom_picture.jpg)


