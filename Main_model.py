import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
from copy import deepcopy
from tqdm import trange
from talib.abstract import *
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from tsfresh.feature_extraction import extract_features, MinimalFCParameters, EfficientFCParameters


df = pd.read_csv("../../paper-related-codev1/data/hs300_domains_v5/保险/sh601318_中国平安_byd.csv")
df.head(3)


# ------ basic module ------
class GELU(nn.Module):  ##（非线性）激活函数   Gaussian Error Linear Units  模型的输入是由非线性激活与随机正则两者共同决定
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5*x*(1+torch.tanh(np.sqrt(2/np.pi)*(x+0.044715*torch.pow(x,3))))

class ModelPatch(nn.Module):   ##patch 是内核kernel的输入
    def __init__(self,in_chs,mid_chs):
        super(ModelPatch,self).__init__()
        
        '''
        nn.Sequential(),一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行
        in_channels:取决于照片的类型，彩色（RGB）为3，灰色为1
        out_channels:取决于过滤器的数量
        一维的时候，filter=kernel
        
        '''
        
        ## 
        self.residual = nn.Sequential(                                             
            nn.Conv1d(in_channels=in_chs, out_channels=mid_chs, kernel_size=1),
            GELU(),
            nn.Conv1d(in_channels=mid_chs,out_channels=in_chs,kernel_size=1)
            )
    def forward(self,x):
        return x+self.residual(x)

class ModelPatch_wo_identify(nn.Module):
    def __init__(self,in_chs,mid_chs):
        super(ModelPatch_wo_identify,self).__init__()
        self.residual = nn.Sequential(
            nn.Conv1d(in_channels=in_chs, out_channels=mid_chs, kernel_size=1),
            GELU(),
            nn.Conv1d(in_channels=mid_chs,out_channels=in_chs,kernel_size=1)
            )
    def forward(self,x):
        return self.net(x)
        
class Chomp1d(nn.Module):  ##裁剪模块
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        # 表示对继承父类属性进行初始化
        self.chomp_size = chomp_size

    def forward(self, x):
        '''
        一个裁剪的模块，裁剪多出来的padding
        tensor.contiguous()会返回有连续内存的相同张量
        有些tensor的view()操作依赖于内存是整块的，这时只需执行
        contiguous()函数，就是把tensor变成在内存中连续分布的形式
        本函数主要是增加padding方式对卷积后的张量做切边而实现因果卷积
        
        '''
        return x[:, :, :-self.chomp_size].contiguous()
    

#两层一维卷积，两层weight_norm,两层chomd1d，非线性激活函数GELU，dropout 为0.2
class TemporalBlock(nn.Module):  ##  pytorch中一切自定义操作都继承自nn.module, 必须在构造函数中执行父类的构造函数
    
#     init 在创建类的时候自动执行，写神经网络的时候一些网络结构的设置最好放在init里
    def __init__(self, n_inputs, n_outputs,     ##类中的第一个函数调用参数必须是self(不能省略)
                 kernel_size=[3,3], stride=[1,1], 
                 dilation=[1,2], 
                 dropout=0.2):
        
        '''
        相当于一个residual block
        n_inputs：int 输入通道数
        n_outputs：int 输出通道数
        kernel_size: int, 卷积核尺寸
        stride：int, 步长
        padding:int, 填充系数
        dilation: int，膨胀系数
        dropout：float, dropout比率
        '''
        
        
        
        super(TemporalBlock,self).__init__()  ## 子类继承父类的init
        # padding can be calculated by the dilation
        
        ## padding=特征图填充宽度 dilation=扩张因子（？） kernel_size=卷积核大小 stride=卷积步长 
        
        ## 卷积前特征宽度N+2*padding = (卷积后特征宽度M-1)*stride+kernel_size
        
        padding = [0,0]
        padding[0] = (kernel_size[0] - 1)*dilation[0]  
        padding[1] = (kernel_size[1] - 1)*dilation[1]
        
        
        
        self.conv1 = nn.Conv1d(n_inputs,n_outputs,kernel_size[0],stride=stride[0],padding=padding[0],dilation=dilation[0])
        
        # 经过conv1，输出的size其实是(batch,input_channel,seq_len+padding)
        
        self.chomp1 = Chomp1d(padding[0])   #裁掉多出来的padding部分，维持输出时间步为seq_len
        self.ln1 = nn.LayerNorm(n_outputs)
        self.gelu1 = GELU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size[1],stride=stride[1], padding=padding[1], dilation=dilation[1])
        self.chomp2 = Chomp1d(padding[1])   #裁掉多出来的padding部分，维持输出时间步为seq_len
        self.ln2 = nn.LayerNorm(n_outputs)
        self.gelu2 = GELU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        
        self.init_weights()
        
    def forward(self,x):
        
        '''
        x: size of (batch,input_channel,seq_len)
        '''
        
        
        # conv1 block
        tmp = self.conv1(x)
        tmp = self.chomp1(tmp) # N,C,L
        tmp = self.ln1(tmp.transpose(1,2).contiguous()) # N,C,L->N,L,C
        tmp = tmp.transpose(1,2).contiguous() # N,L,C->N,C,L
        tmp = self.
        
        
        1(tmp)
        tmp = self.dropout1(tmp)
        
        
        # conv2 block
        tmp = self.conv2(tmp)
        tmp = self.chomp2(tmp)
        tmp = self.ln2(tmp.transpose(1,2).contiguous()) # N,C,L->N,L,C
        tmp = tmp.transpose(1,2).contiguous() # N,L,C->N,C,L
        tmp = self.gelu2(tmp)
        tmp = self.dropout2(tmp)
        
        out = tmp
        res = x if self.downsample is None else self.downsample(x)
        return out+res
    
    def init_weights(self):
        '''
        参数初始化
        
        '''
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)


            
#时序卷积模块，使用for循环对8层隐含层，每层25个节点进行构建。其中*layer 表示迭代器拆分layers为一层层网络            
class TemporalConvNet(nn.Module):
    def __init__(self,num_inputs,num_channels,
                 model_patch=ModelPatch, # model_patch should be a nn.Module
                 dropout=0.2,
                 ):
        """
        
        TCN,目前paper给出的TCN结构很好的支持每个时刻一个数的情况，即sequence 结构
        对于每个时刻为一个向量的一维结构，勉强可以把向量拆成若干该时刻的输入通道
        对于每个时刻为一个矩阵或更高维图像的情况不太好办
        
        num_inputs：int 输入通道数
        num_channels:list,每层的hidden_channel数，i.e. [25,25,25,25]表示有4个隐层，每层hidden_channel数为25
        dropout：float, dropout比率
        """
        
        
        super().__init__()
        # initilize the _tl with False value
        self._tl = False 
        
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = [1,2] ##为什么膨胀系数是一个向量，且与i无关？
            in_channels = num_inputs if i==0 else num_channels[i-1]  ##确定每一层的输入通道数，输入层通道为1，隐含层...
            out_channels = num_channels[i]   ## 确定每一层的输出通道数
            layers += [TemporalBlock(in_channels, out_channels, kernel_size=[3,3], stride=[1,1], dilation=dilation_size,dropout=dropout)]
        
        self.layers = nn.ModuleList(layers) 
        
        # then we build the path with model_patch
        # self.model_patchs = OrderedDict()
        
        self.model_patchs = nn.ModuleDict()
        for i in range(num_levels):
            for j in range(i+1,num_levels):
                if j!=i+1:
                    patch_name = f"{i}t{j}"
                    #print(patch_name)
                    self.model_patchs[patch_name] = model_patch(in_chs=num_channels[i],mid_chs=num_channels[i]//2)  
        
        # linear layer
        self.linear = nn.Linear(num_channels[-1], 1)
        return
    
    def _set2tl(self):
        self._tl = True
        for param in self.layers.parameters():
            param.requires_grad = False
        for param in self.linear.parameters():
            param.requires_grad = False
        for param in self.model_patchs.parameters():
            param.requires_grad = True
        
    def _set2dl(self):
        self._tl = False
        for param in self.layers.parameters():
            param.requires_grad = True
        for param in self.linear.parameters():
            param.requires_grad = True
        for param in self.model_patchs.parameters():
            param.requires_grad = True
    
    def forward(self,x):
        
        if self._tl == False:
            input_ = x
            for idx,layer in enumerate(self.layers,0):
                output_ = layer(input_)
                input_ = output_
            
            output_ = self.linear(input_[:,:,-1])
            # return output_
            return torch.sigmoid(output_)
            
        output_list = []
        for idx,layer in enumerate(self.layers,0):
            # print(f"This is {idx} layer ")
            if idx == 0:
                input_ = x
            else:
                input_ = 0
                for i in range(idx):
                    if i!= idx-1:
                        patch_name = f"{i}t{idx}"
                        #print(patch_name)
                        #print(output_list[i].shape)
                        delta_ = self.model_patchs[patch_name](output_list[i])
                        #print(delta_)
                        #print("delta_")
                        #print(delta_.shape)
                        # here, we use addition operation as the fusion method
                        # if the fusion method is changed to cat, then torch.cat should be used here.
                        input_ = input_ + delta_
                    else:
                        input_ = input_ + output_list[i]
                    
            output_ = layer(input_)
            output_list.append(output_)
        
        output_ = self.linear(output_[:,:,-1])
        
        # return output_
        return torch.sigmoid(output_)
  
  
  

'''

'''



def get_tsfeatures(df, use_volume=False, dropna=True, scaled=True, mode="Minimal"):
    window_size = 30
    
    ## 将30天内的date和stock_id 合并项 当前日期到30天后每天都有
    dfs = []
    for i in range(window_size, len(df)):
        date = df.iloc[i, 0]
        sdf = df.iloc[i-window_size:i].copy()
        sdf['code'] = sdf['code'] + "_" + date
        dfs.append(sdf)
    dfm = pd.concat(dfs) 
    
    ##选取变量
    use_cols = ['code', 'date', 'open', 'high', 'low', 'close', 'preclose', 'volume', 'amount']
    if not use_volume:
        use_cols.remove("volume")
    df_feature = dfm[use_cols]
    
    ##计算同一个code的特征的sum、mean、median、length：30, std、 var, root_mean_square、maximum、absolute_maximum、minimum
    settings = MinimalFCParameters() if mode == "Minimal" else EfficientFCParameters()
    features = extract_features(df_feature, column_id="code", column_sort="date", default_fc_parameters=settings)
    features = features.reset_index()
    
    
    ##把合成的新code(现index) 拆成code和date
    features['code'] = features['index'].str.split("_").str[0]
    features['date'] = features['index'].str.split("_").str[1]
    
    
    dff = pd.merge(df[use_cols], features.drop(columns="index"), on=['code', 'date'], how="left")
    drop_cols = [col for col in dff.columns if col.endswith("length")]
    dff = dff.drop(columns=drop_cols)
    
    
    if dropna:
        dff = dff.dropna(axis=1, thresh=int(0.9*len(dff))).dropna()
    dff.fillna(0, inplace=True)
    if scaled:
        dff.iloc[:, 2:] = MinMaxScaler().fit_transform(dff.iloc[:, 2:])
    return dff



def get_Technical_Indicator(df, timeperiod=14, use_volume=False, dropna=True, scaled=True):
    drop_cols = ["adjustflag", "turn", "tradestatus", "pctChg", "isST"]
    if not use_volume:
        drop_cols.append("volume")
    df = df.sort_values("date")
    
    df['adx'] = ADX(df['high'], df['low'], df['close'], timeperiod=timeperiod)
    
    df['apo'] = APO(df['close'], fastperiod=12, slowperiod=26, matype=0)
    
    df['bop'] = BOP(df['open'], df['high'], df['low'], df['close'])
    
    df['mfi'] = MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=timeperiod)
    
    df['mom'] = MOM(df['close'], timeperiod=timeperiod)
    
    df['ppo'] = PPO(df['close'], fastperiod=12, slowperiod=26, matype=0)
    
    df['rsi'] = RSI(df['close'], timeperiod=timeperiod)

    fastk, fastd = STOCHF(df['high'], df['low'], df['close'], fastk_period=5, fastd_period=3, fastd_matype=0)
    df['fastk'] = fastk
    df['fastd'] = fastd

    slowk, slowd = STOCH(df['high'], df['low'], df['close'], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    df['slowk'] = slowk
    df['slowd'] = slowd

    df['willr'] = WILLR(df['high'], df['low'], df['close'], timeperiod=timeperiod)

    macd, macdsignal, macdhist = MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd'] = macd

    df['roc_close'] = ROC(df['close'], timeperiod=timeperiod)

    df['cci'] = CCI(df['high'], df['low'], df['close'], timeperiod=timeperiod)

    if use_volume:
        df['obv'] = OBV(df['close'], df['volume'])
    
    df['p1ccr'] = df['close'] / df['close'].shift(1) - 1
    df['p2ccr'] = df['close'] / df['close'].shift(2) - 1
    df['p3ccr'] = df['close'] / df['close'].shift(3) - 1
    df['p4ccr'] = df['close'] / df['close'].shift(4) - 1
    df['p5ccr'] = df['close'] / df['close'].shift(5) - 1
    
    df = df.drop(columns=drop_cols)
    if dropna:
        df.dropna(inplace=True)
    if scaled:
        df.iloc[:, 2:] = MinMaxScaler().fit_transform(df.iloc[:, 2:])
    return df
    
   
   
   
def get_dataset(df):
    features = []
    labels = []
    window = 30
    for i in range(window, len(df)):
        feature = df.iloc[i-window:i, 2:].values.T  # C * L
        ratio = df['close'].iloc[i] / (df['close'].iloc[i-1] + 1e-9)
        label = 1 if ratio >= 1 else 0
        features.append(feature)
        labels.append(label)
        
    features = torch.tensor(features, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.float)
    
    N1 = int(len(features)*0.7)
    N2 = int(len(features)*0.8)
    N = len(features)
    idx_train = list(range(N1))
    idx_val = list(range(N1, N2))
    idx_test = list(range(N2, N))
    X_train, y_train = features[idx_train], labels[idx_train]
    X_val, y_val = features[idx_val], labels[idx_val]
    X_test, y_test = features[idx_test], labels[idx_test]
    
    return X_train, X_val, X_test, y_train, y_val, y_test
    
    
    
    
def cal_acc(pred, label):
    pred = (pred.squeeze()>0.5).float()
    acc = (pred == label).float().mean()
    return acc
    
    
    
    
def train(df, file=None, num_channels = [32, 32, 32, 32, 32]):
    # datasets
    X_train, X_val, X_test, y_train, y_val, y_test = get_dataset(df)
    
    '''
    一个epoch(代)是指整个数据集正向反向训练一次。它被用来提示模型的准确率并且不需要额外数据。
    '''
    
    
    train_batch_size = 32  ##batch-size 即 一次训练所抓取的数据样本数量，batch-size大小影响训练速度和模型优化，同样影响 epoch训练模型次数
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=train_batch_size, shuffle=True) ## DataLoader本质上是一个iterable（跟python内置类型list等一样），并利用多进程；来加速batch data的处理
                                                                                                          # TensorDataset 对tensor进行打包

    # models
    num_inputs = X_train.shape[1]
    
    model = TemporalConvNet(num_inputs, num_channels)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)  ##使用Adam优化器，lr=学习率，默认为 1e-3
    criterion = nn.BCELoss() ##计算目标值和预测值之间的二进制交叉损失函数  loss=-w*[p*log(q)+(1-p)*log(1-q))]
    
    # training
    best_acc = 0.0
    best_model = None
    logs = {}
    for epoch in trange(50):    ###trange(i) 是tqdm(range(i)) 迭代进度条
        for x, y in train_loader:
            y_pred = model(x)
            y_pred = y_pred.view(-1)
            loss = criterion(y_pred, y)
            optimizer.zero_grad() ## 把模型参数梯度设为0
            loss.backward()
            optimizer.step()
        y_train_pred = model(X_train)
        acc_train = cal_acc(y_train_pred, y_train)
        y_val_pred = model(X_val)
        acc_val = cal_acc(y_val_pred, y_val)
        if acc_val > best_acc:
            best_acc = acc_val
            best_model = deepcopy(model)
        logs[epoch] = {"train_loss":loss.item(), "acc_train":acc_train.item(), "acc_val":acc_val.item()}
    
    # metrics on test dataset
    pred_y_test = best_model(X_test)
    acc_test = cal_acc(pred_y_test, y_test)
    print(f"file: {file}, acc on test: {acc_test}")
    df_log = pd.DataFrame(logs).T
    return acc_test, df_log
    
    
    
    
files = {
    "sz002179_中航光电":"data/sz002179_中航光电_byd.csv",
    "sh601601_中国太保":"data/sh601601_中国太保_byd.csv",
    "sh600029_南方航空":"data/sh600029_南方航空_byd.csv",
    "sh601808_中海油服":"data/sh601808_中海油服_byd.csv",
    "sh601818_光大银行":"data/sh601818_光大银行_byd.csv",
    "sh601377_兴业证券":"data/sh601377_兴业证券_byd.csv"
}




fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for i, (file, path) in enumerate(files.items()):
    df = pd.read_csv(path)
    dfti = get_Technical_Indicator(df)
#     dfts = get_tsfeatures(df)
    _, df_log = train(dfti, file)
    df_log.plot(title=file, ax=axes.flatten()[i])
    
    
    
    
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for i, (file, path) in enumerate(files.items()):
    df = pd.read_csv(path)
#     dfti = get_Technical_Indicator(df)
    dfts = get_tsfeatures(df)
    _, df_log = train(dfts, file)
    df_log.plot(title=file, ax=axes.flatten()[i])
    
    
    
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for i, (file, path) in enumerate(files.items()):
    df = pd.read_csv(path)
#     dfti = get_Technical_Indicator(df)
    dfts = get_tsfeatures(df, mode="Efficient")
    _, df_log = train(dfts, file)
    df_log.plot(title=file, ax=axes.flatten()[i])
    
    
    
    
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for i, (file, path) in enumerate(files.items()):
    df = pd.read_csv(path)
    dfti = get_Technical_Indicator(df)
    dfts = get_tsfeatures(df)
    redundant_cols = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount']
    sel_cols = [col for col in dfts.columns if col not in redundant_cols]
    dff = pd.merge(dfti, dfts[sel_cols])
    num_channels = [64, 32, 32, 32, 32]
    _, df_log = train(dff, file, num_channels)
    df_log.plot(title=file, ax=axes.flatten()[i])
    
    
    
    
    
    
    
    
    
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for i, (file, path) in enumerate(files.items()):
    df = pd.read_csv(path)
    dfti = get_Technical_Indicator(df)
    dfts = get_tsfeatures(df, mode="Efficient")
    redundant_cols = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount']
    sel_cols = [col for col in dfts.columns if col not in redundant_cols]
    dff = pd.merge(dfti, dfts[sel_cols])
    _, df_log = train(dff, file)
    df_log.plot(title=file, ax=axes.flatten()[i])
    
    
    
    
files = os.listdir("data/")
scores = {}



for file in files:
    df = pd.read_csv("data/"+file)
    dfti = get_Technical_Indicator(df)
    dfts = get_tsfeatures(df)
    redundant_cols = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount']
    sel_cols = [col for col in dfts.columns if col not in redundant_cols]
    dff = pd.merge(dfti, dfts[sel_cols])
    acc_test, _ = train(dff, file)
    scores[file[:-8]] = acc_test.item()
    
    
acc_scores = pd.Series(scores)
acc_scores.describe()
acc_scores[acc_scores>0.55].sort_values(ascending=False)
acc_scores[acc_scores>0.55].sort_values(ascending=True).plot(kind='barh')
acc_scores.hist(bins=30)
sr[sr>0.55].plot(kind="barh")
