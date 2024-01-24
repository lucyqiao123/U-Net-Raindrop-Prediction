import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch. nn import functional as F
from visdom import Visdom
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal as MVN
from torch.nn.modules.loss import _Loss
viz = Visdom()#创建窗口
viz. line([[0.0,0.0]], [0.0], win='train&test loss epoch', opts=dict(title='train&test loss epoch',legend=['train',"test"]))
viz. line([[0.0,0.0]], [0.0], win='train loss&csi batch', opts=dict(title='train loss&csi batch',legend=['loss',"csi"]))
viz. line([0.0], [0.0], win='test csi batch', opts=dict(title='test csi batch',legend=['test']))
class down_ResBlk(nn. Module): #encoder 残差块
    def __init__(self, ch_in, ch_out): #此处通道数为10不变 仅实现卷积以及标准化
        super(down_ResBlk,self).__init__()
        self. conv1= nn. Conv2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1)
        self. conv2 = nn. Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self. bn1 = nn. BatchNorm2d(ch_out)
        self.conv3 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.extra = nn. Sequential(
            nn. Conv2d(ch_out, ch_out, kernel_size=1, stride=1),#这里stride要同步
            nn. BatchNorm2d(ch_out))#short_cut 创造一个模块

    def forward(self, x):
        x2 = F.relu(self.conv1(x))
        x3 = self.bn1(F.leaky_relu(self.conv2(x2)))
        x3 = self.conv3(x3)
        # print("shape x3",x3.shape)
        # print("shape x3", self.extra(x2).shape)
        # short cut.
        x3 = self.extra(x2)+x3
        # print(out.shape)
        return x3
class up_ResBlk(nn. Module): #decoder 残差块
    def __init__(self, ch_in, ch_out): #此处通道数为10不变 仅实现卷积以及标准化
        super(up_ResBlk,self).__init__()
        self.up_sample = nn.Upsample(scale_factor=2)
        self. conv1= nn. Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1)
        self. conv2 = nn. Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self. bn1 = nn. BatchNorm2d(ch_out)
        self.conv3 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.extra = nn. Sequential(
            nn. Conv2d(ch_out, ch_out, kernel_size=1, stride=1),#这里stride要同步
            nn. BatchNorm2d(ch_out))#short_cut 创造一个模块

    def forward(self, x):
        x2 = F.relu(self.conv1(self.up_sample(x)))
        x3 = self.bn1(F.leaky_relu(self.conv2(x2)))
        x3 = self.conv3(x3)
        # print("shape x3",x3.shape)
        # print("shape x3", self.extra(x2).shape)
        # short cut.
        x3 = self.extra(x2)+x3
        # print(out.shape)
        return x3
class SE_Block(nn.Module):#注意力机制
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应平均池化，括号里是输出的h,w尺寸
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x): # x: [batch_size,512*3,2,2]
        b, c, _, _ = x.size() #
        y = self.avg_pool(x).view(b, c)  # squeeze操作 -->[batch_size,512*3,1,1]-->[batch_size,512*3]
        y = self.fc(y).view(b, c, 1, 1)  # FC获取通道注意力权重，是具有全局信息的 [batch_size,512*3,1,1]
        out = x * y.expand_as(x)
        return out  # 注意力作用每一个通道上
class skip_conv(nn.Module):#跳跃连接的卷积层
    def __init__(self, in_ch, out_ch):
        super(skip_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x
class Encoder(nn. Module): #encoder 7层包装起来
    def __init__(self, img_ch=10):
        super().__init__()
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8]

        self.d1 = down_ResBlk(img_ch, n1)
        self.d2 = down_ResBlk(n1, filters[1])
        self.d3 = down_ResBlk(filters[1], filters[2])
        self.d4 = down_ResBlk(filters[2], filters[3])
        self.d5 = down_ResBlk(filters[3], filters[3])
        self.d6 = down_ResBlk(filters[3], filters[3])
        self.d7 = down_ResBlk(filters[3], filters[3])

    def forward(self, x):
        #x: [b, 10, 256, 256]
        e1 = self.d1(x)      # [b,64,128,128]
        e2 = self.d2(e1)     # [b,128,64,64]
        e3 = self.d3(e2)     # [b,256,32,32]
        e4 = self.d4(e3)     # [b,512,16,16]
        e5 = self.d5(e4)     # [b,512,8,8]
        e6 = self.d6(e5)     # [b,512,4,4]
        e7 = self.d7(e6)     # [b,512,2,2]
        return e1, e2, e3, e4, e5, e6, e7


class UNet(nn.Module):
    def __init__(self, img_ch, final_ch):
        super(UNet, self).__init__()
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8]  # [64, 128, 256, 512]

        self.Encoder_dBR = Encoder(img_ch)
        self.Encoder_ZDR = Encoder(img_ch)
        self.Encoder_KDP = Encoder(img_ch)

        self.SE_Block = SE_Block(filters[3]*3)
        self.skip_Block=skip_conv(filters[3]*6,filters[3]*3)

        self.u7 = up_ResBlk(filters[3]*3 , filters[3])
        self.skip7 = skip_conv(filters[3]*4,filters[3])

        self.u6 = up_ResBlk(filters[3], filters[3])
        self.skip6 = skip_conv(filters[3]*4, filters[3])

        self.u5 = up_ResBlk(filters[3], filters[3])
        self.skip5 = skip_conv(filters[3] * 4, filters[3])

        self.u4 = up_ResBlk(filters[3], filters[2])
        self.skip4 = skip_conv(filters[2] * 4, filters[2])

        self.u3 = up_ResBlk(filters[2], filters[1])
        self.skip3 = skip_conv(filters[1] * 4, filters[1])

        self.u2 = up_ResBlk(filters[1], filters[0])
        self.skip2 = skip_conv(filters[0] * 4, filters[0])

        self.u1= up_ResBlk(filters[0], final_ch)
        self.active = torch.nn.ReLU()

    def forward(self, dBR, ZDR, KDP):
        #'Zh, Zdr, Kdp: [batch_size, 10, 256, 256]'
        #7 layers encoder-->[batch_size,512,2,2]
        encoder_dBR = self.Encoder_dBR(dBR) #7层数据
        encoder_ZDR = self.Encoder_dBR(ZDR)
        encoder_KDP = self.Encoder_dBR(KDP)
        # print(encoder_dBR.shape)
        merge7= torch.concat((encoder_dBR[6], encoder_ZDR[6], encoder_KDP[6]), dim=1)   # [bsz, 512*3, 2, 2] 直接cat得到的数据
        u_Block=self.SE_Block(merge7) #[bsz, 512*3, 2, 2] 加注意力后得到的数据
        u_Block=torch.concat((merge7,u_Block), dim=1) #[bsz, 512*6, 2, 2]
        u_Block=self.skip_Block(u_Block) # [bsz, 512*3, 2, 2]

        u7=self.u7(u_Block) # [bsz, 512, 4, 4]
        u7 = torch.concat((encoder_ZDR[5], encoder_KDP[5], encoder_dBR[5], u7), dim=1) #[bsz, 512*4, 4, 4]
        u7 = self.skip7(u7) # [bsz, 512, 4, 4]

        u6=self.u6(u7) # [bsz, 512, 8, 8]
        u6 = torch.concat((encoder_ZDR[4], encoder_KDP[4], encoder_dBR[4], u6), dim=1) #[bsz, 512*4, 8, 8]
        u6 = self.skip6(u6) # [bsz, 512, 8, 8]

        u5=self.u5(u6) # [bsz, 512, 16, 16]
        u5 = torch.concat((encoder_ZDR[3], encoder_KDP[3], encoder_dBR[3], u5), dim=1) #[bsz, 512*4, 16, 16]
        u5 = self.skip5(u5) # [bsz, 512, 16, 16]

        u4=self.u4(u5) # [bsz, 256, 32, 32]
        u4 = torch.concat((encoder_ZDR[2], encoder_KDP[2], encoder_dBR[2], u4), dim=1) #[bsz, 256*4, 32, 32]
        u4 = self.skip4(u4) # [bsz, 256, 32, 32]

        u3=self.u3(u4) # [bsz, 128, 64, 64]
        u3 = torch.concat((encoder_ZDR[1], encoder_KDP[1], encoder_dBR[1], u3), dim=1) #[bsz, 128*4, 64, 64]
        u3 = self.skip3(u3) # [bsz, 128, 64, 64]

        u2=self.u2(u3) # [bsz, 64, 128, 128]
        u2 = torch.concat((encoder_ZDR[0], encoder_KDP[0], encoder_dBR[0], u2), dim=1) #[bsz, 64*4, 128, 128]
        u2 = self.skip2(u2) # [bsz, 64, 128, 128]

        u1=self.u1(u2) # [bsz, 10, 256, 256]
        # print(decoder_dBR.shape)
        out = self.active(u1)
        # print(out.shape)

        return out

class MyDataset(Dataset):
    def __init__(self,vars,mode):
        self.data_path=r'C:\Users\llush\Desktop\数模re\数模_原\模型实现代码附件\data_after_wash_new'
        self.vars=vars
        self.mode=mode
        if self.mode=="train":
            self.sample_start=200 #开始的文件名 选取90%的作为训练样本
            self.sample_end=9400
        elif self.mode=="test":
            self.sample_start=9600
            self.sample_end=10600
        self.list_idx=[i for i in range(self.sample_start,self.sample_end+1,200)]

    def __len__(self):
        if self.mode=="train":
            length=self.sample_end-self.sample_start+200
        else:
            # length = self.sample_end - self.sample_start + 200
            length=10740-self.sample_start
        return length

    def __getitem__(self, idx):
        sample_idx = self.list_idx[int(idx/200)]
        data_x={}
        data_y={}
        for var in self.vars:#一次包装三个变量
            path = self.data_path + "\{}_sample{}.npy".format(var, sample_idx)  # [200,20,256,256]
            data_x[var]= torch.tensor(np.load(path)[:, :10, :, :][idx - sample_idx + self.sample_start])  # 取前十帧 [200,10,256,256]
            data_y[var]= torch.tensor(np.load(path)[:, 10:, :, :][idx - sample_idx + self.sample_start])
            # print(data_x[var].shape)
            # print(data_y[var].shape)
        #为了方便shuffle=True打乱 随用随取
        # print("idx",idx)
        # print("sample_idx",sample_idx)
        return data_x,data_y


batch_size=4
dBZ_KDP_ZDR_train_dataset = MyDataset(vars=["dBZ", "KDP", "ZDR"], mode="train")
dBZ_KDP_ZDR_test_dataset = MyDataset(vars=["dBZ", "KDP", "ZDR"], mode="test")
dBZ_KDP_ZDR_train_loader = DataLoader(dBZ_KDP_ZDR_train_dataset, batch_size=batch_size, shuffle=True)
dBZ_KDP_ZDR_test_loader = DataLoader(dBZ_KDP_ZDR_test_dataset, batch_size=batch_size, shuffle=False)
# for i, data_batch in enumerate(dBZ_KDP_ZDR_train_loader):
#     print(len(data_batch[0]),data_batch[0]["dBZ"].shape)
#     print(len(data_batch[1]),data_batch[0]["KDP"].shape)
#     break
def draw(frames_true,frames_pred): #test画图部分
    # 绘制前10帧和后10帧的图片
    num_rows = 4
    num_cols = 5
    num_images = num_rows * num_cols
    sample_true=frames_true[-1] #[4,10,256,256]-->[10,256,256]
    sample_pred=frames_pred[-1]
    plt.figure(figsize=(12, 10))
    for i in range(num_images):
        plt.subplot(num_rows, num_cols, i + 1)
        if i < 10:
            plt.imshow(sample_true[i], cmap='coolwarm')  # 更改cmap以使用不同的颜色图
            plt.title(f'Frame {i + 1} (True)')
        else:
            plt.imshow(sample_pred[i - 10], cmap='coolwarm')  # 更改cmap以使用不同的颜色图
            plt.title(f'Frame {i - 9} (Pred)')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def prep_clf(obs, pre, threshold=35):
    # 根据阈值分类为 0, 1
    obs = np.where(obs >= threshold, 1, 0)
    pre = np.where(pre >= threshold, 1, 0)
    # True positive (TP)
    hits = np.sum((obs == 1) & (pre == 1))
    # False negative (FN)
    misses = np.sum((obs == 1) & (pre == 0))
    # False positive (FP)
    falsealarms = np.sum((obs == 0) & (pre == 1))
    # True negative (TN)
    correctnegatives = np.sum((obs == 0) & (pre == 0))
    return hits, misses, falsealarms, correctnegatives


def CSI(frames_true, frames_pred, threshold=35):#[4,10,256,256]
    csi_res=[]
    for i in range(4):
        for j in range(10):
            hits, misses, falsealarms, correctnegatives = prep_clf(obs=frames_true[i][j], pre=frames_pred[i][j], threshold=threshold)
            if hits + falsealarms + misses==0:
                csi_res.append(0)
            else:
                csi_res.append(hits / (hits + falsealarms + misses))
    csi=np.max(np.array(csi_res))#取最大的一帧csi
    return csi

# 定义Balanced MSE Loss（BMC版本）多变量版本
def reshape_for_bmc_loss(pred, target):
    device = torch.device('cuda')
    pred = F.interpolate(pred, scale_factor=0.125).to(device)
    target = F.interpolate(target, scale_factor=0.125).to(device)
    batch_size = pred.size(0)
    pred = pred.view(batch_size, -1)
    target = target.view(batch_size, -1)
    return pred, target

def bmc_loss_md(pred, target, noise_var):
    device = torch.device('cuda')
    pred, target = reshape_for_bmc_loss(pred, target)
    I_diag = torch.ones(pred.shape[-1], device=device)
    logits = MVN(pred.unsqueeze(1), noise_var * torch.diag(I_diag)).log_prob(target.unsqueeze(0)).to(device)
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]).to(device)).to(device)
    loss = loss * (2 * noise_var).detach().to(device)
    return loss

class BMCLoss(_Loss):
    def __init__(self, init_noise_sigma):
        super(BMCLoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma))
    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        return bmc_loss_md(pred, target, noise_var)




def main():

    device = torch.device('cuda')
    model = UNet(10, 10).to(device)
    init_noise_sigma = 8.0
    sigma_lr = 1e-2
    criterion = nn.MSELoss().to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.03)
    # optimizer.add_param_group({'params': criterion.noise_sigma, 'lr': sigma_lr, 'name': 'noise_sigma'})
    # criterion = nn.MSELoss().to(device)

    print(model)
    label1=0
    label2 = 0
    for epoch in range(10):
        model.train()
        for i, data_batch in enumerate(dBZ_KDP_ZDR_train_loader):
            # (dBZ_x, KDP_x, ZDR_x)
            dBZ_x, KDP_x, ZDR_x = (
                data_batch[0]["dBZ"].to(device),
                data_batch[0]["KDP"].to(device),
                data_batch[0]["ZDR"].to(device),
            )
            # 目标tatget dBZ_y
            dBZ_y = data_batch[1]["dBZ"].to(device)
            logits = model(dBZ_x, KDP_x, ZDR_x)
            loss = criterion(logits,dBZ_y) #Balanced MSE 需要压缩维度[batch,10*256*256]
            train_loss = loss.item()
            dBZ_y = dBZ_y * 65  # 为了计算csi指标而从标准化转换回去 [4,10,256,256]
            logits = logits * 65
            train_csi = CSI(dBZ_y.cpu(), logits.cpu())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            viz.line([[train_loss,train_csi]], [label1], win='train loss&csi batch', update='append')
            label1+=1
            print("epoch:", epoch,i, "train",train_loss)

        # test 模式
        model.eval()
        with torch.no_grad():  # 声明不需要进行forward计算梯度，避免不必要的梯度混淆的麻烦
            test_loss=[]

            for i, data_batch in enumerate(dBZ_KDP_ZDR_test_loader):

                # data_batch --> data_x: [b, 10, 256, 256]  data_y: [b, 10, 256, 256]
                dBZ_x, KDP_x, ZDR_x = (
                    data_batch[0]["dBZ"].to(device),
                    data_batch[0]["KDP"].to(device),
                    data_batch[0]["ZDR"].to(device),
                )
                # 目标tatget dBZ_y
                dBZ_y = data_batch[1]["dBZ"].to(device)
                logits = model(dBZ_x, KDP_x, ZDR_x)
                loss = criterion(logits, dBZ_y)
                test_loss.append(loss.item())
                dBZ_y=dBZ_y*65 #为了计算csi指标而从标准化转换回去 [4,10,256,256]
                logits=logits*65 #[4,10,256,256]
                csi=CSI(dBZ_y.cpu(),logits.cpu()) #计算指标
                viz.line([csi], [label2], win='test csi batch',update='append')
                label2+=1
                print("epoch:", epoch,i, "test",csi)
            draw(dBZ_y.cpu(),logits.cpu()) #加.cpu()是将张量转换到cpu上，不然不能进行numpy转换计算
        viz.line([[train_loss, np.array(test_loss).mean()]], [epoch], win='train&test loss epoch', update='append')


#
main()
#
# batch_size=4 #样本数
# # 构建数据集
#
# # for i,(x,y) in enumerate(train_loader):
# #     print(x.shape)
# #     print(y.shape)
#
# # dBR=torch.randn((4, 10, 256, 256))
# # ZDR=torch.randn((4, 10, 256, 256))
# # KDP=torch.randn((4, 10, 256, 256))
# # UNet(10,10)(dBR,ZDR,KDP)
#
#

# main()

