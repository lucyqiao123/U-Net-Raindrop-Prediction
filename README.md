# U-Net Raindrop Prediction
复刻论文：Improving Nowcasting of Convective Development by Incorporating Polarimetric Radar Variables Into a Deep‐Learning Model.Geophysical Research Letters,48(21), e2021GL095302

数据：论文supporting information/2023年研究生数模大赛F题官网

技术文件路径：data_wash_label_index.py--->产出label_index.npy-->var_data_filter_re.py存储样本数据-->U_net.py

目前存在问题：复刻论文提到使用B-MSE，作者内存不够没用上；train、test样本分布不均衡，训练效果由样本均衡性决定，换一批样本test效果就有所变化


