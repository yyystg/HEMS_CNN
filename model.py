import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CNN_HEMS(nn.Module):

    def __init__(self, args):
        super(CNN_HEMS, self).__init__()
        self.args = args

        V = args.input_dim
        D = args.input_num
        C = args.output_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes
        self.Cdepth=args.conv_depth
        Fs = args.fc_size

        self.embed = nn.Embedding(V, D)
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.conv1_bn = nn.BatchNorm2d(Co)
        if self.Cdepth >= 2:
            self.convs2 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, Co)) for K in Ks])
            self.convs2_bn = nn.BatchNorm2d(Co)
        if self.Cdepth >= 3:
            self.convs3 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, Co)) for K in Ks])
            self.convs3_bn = nn.BatchNorm2d(Co)
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        # self.dropout = nn.Dropout(args.dropout)
        if Fs[0]==-1:
            self.fc1 = nn.Linear(len(Ks) * Co, C)
            self.fc1_bn = nn.BatchNorm1d(C)
        else:
            if len(Fs)>=1:
                self.fc1 = nn.Linear(len(Ks)*Co,Fs[0])
                self.fc1_bn = nn.BatchNorm1d(Fs[0])
            if len(Fs)>=2:
                self.fc2 = nn.Linear(Fs[0],Fs[1])
                self.fc2_bn = nn.BatchNorm1d(Fs[1])
            self.fc_end = nn.Linear(Fs[-1],C)
            self.fc_end_bn = nn.BatchNorm1d(C)


        for p in self.parameters():
            torch.nn.init.normal_(p,0,std=0.01)

    # def conv_and_pool(self, x, conv):
    #     x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
    #     x = F.avg_pool1d(x, x.size(2)).squeeze(2)
    #     return x

    def forward(self, x):

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        if self.args.batch_normalization==True:
            if self.Cdepth==1:
                x = [F.relu(self.conv1_bn(conv(x))).squeeze(3) for conv in self.convs1]
            elif self.Cdepth==2:
                x = [torch.transpose(F.relu(self.conv1_bn(conv(x))).squeeze(3), 1, 2) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
                x = [F.relu(self.convs2_bn(conv(x.unsqueeze(1)))).squeeze(3) for (conv, x) in zip(self.convs2, x)]
            elif self.Cdepth ==3:
                x = [torch.transpose(F.relu(self.conv1_bn(conv(x))).squeeze(3), 1, 2) for conv in self.convs1]
                x = [torch.transpose(F.relu(self.convs2_bn(conv(x.unsqueeze(1)))).squeeze(3),1,2) for (conv, x) in zip(self.convs2, x)]
                x = [F.relu(self.convs3_bn(conv(x.unsqueeze(1)))).squeeze(3) for (conv, x) in zip(self.convs3, x)]
        else:
            if self.Cdepth==1:
                x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
            elif self.Cdepth==2:
                x = [torch.transpose(F.relu(conv(x)).squeeze(3), 1, 2) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
                x = [F.relu(conv(x.unsqueeze(1))).squeeze(3) for (conv, x) in zip(self.convs2, x)]
            elif self.Cdepth ==3:
                x = [torch.transpose(F.relu(conv(x)).squeeze(3), 1, 2) for conv in self.convs1]
                x = [torch.transpose(F.relu(conv(x.unsqueeze(1))).squeeze(3), 1, 2) for (conv, x) in zip(self.convs2, x)]
                x = [F.relu(conv(x.unsqueeze(1))).squeeze(3) for (conv, x) in zip(self.convs3, x)]

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        # x = self.dropout(x)  # (N, len(Ks)*Co)
        if self.args.batch_normalization == True:
            if self.args.fc_size[0]==-1:
                logit = self.fc1_bn(self.fc1(x))  # (N, C)
            else:
                x = F.relu(self.fc1_bn(self.fc1(x)))
                if len(self.args.fc_size)>=2:
                    x = F.relu(self.fc2_bn(self.fc2(x)))
                logit = self.fc_end_bn(self.fc_end(x))
        else:
            if self.args.fc_size[0]==-1:
                logit = self.fc1(x)  # (N, C)
            else:
                x = self.fc1(x)
                if len(self.args.fc_size)>=2:
                    x = self.fc2(x)
                logit = self.fc_end(x)

        return logit
