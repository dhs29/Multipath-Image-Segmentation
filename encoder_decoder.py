import torch
import os
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
#import nibabel as nib
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv3d = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size,
                                stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(num_features=out_channels)

    def forward(self, x):
        x = self.batch_norm(self.conv3d(x))
        #x = self.conv3d(x)
        x = F.elu(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, model_depth=4, pool_size=2):
        super(EncoderBlock, self).__init__()
        self.root_feat_maps = 4
        self.num_conv_blocks = 2
        # self.module_list = nn.ModuleList()
        self.module_dict = nn.ModuleDict()
        for depth in range(model_depth):
            feat_map_channels = 2 ** (depth + 1) * self.root_feat_maps
            for i in range(self.num_conv_blocks):
                # print("depth {}, conv {}".format(depth, i))
                if depth == 0:
                    # print(in_channels, feat_map_channels)
                    self.conv_block = ConvBlock(in_channels=in_channels, out_channels=feat_map_channels)
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv_block
                    in_channels, feat_map_channels = feat_map_channels, feat_map_channels * 2
                else:
                    # print(in_channels, feat_map_channels)
                    self.conv_block = ConvBlock(in_channels=in_channels, out_channels=feat_map_channels)
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv_block
                    in_channels, feat_map_channels = feat_map_channels, feat_map_channels * 2
            if depth == model_depth - 1:
                break
            else:
                self.pooling = nn.MaxPool3d(kernel_size=pool_size, stride=2, padding=0)
                self.module_dict["max_pooling_{}".format(depth)] = self.pooling

    def forward(self, x):
        down_sampling_features = []
        for k, op in self.module_dict.items():
            if k.startswith("conv"):
                x = op(x)
                #print(k, x.shape)
                if k.endswith("1"):
                    down_sampling_features.append(x)
            elif k.startswith("max_pooling"):
                x = op(x)
                #print(k, x.shape)

        return x, down_sampling_features


class ConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=(3,4,3), stride=2, padding=1, output_padding=1):
        super(ConvTranspose, self).__init__()
        self.conv3d_transpose = nn.ConvTranspose3d(in_channels=in_channels,
                                                   out_channels=out_channels,
                                                   kernel_size=k_size,
                                                   stride=stride,
                                                   padding=padding,
                                                   output_padding=output_padding)

    def forward(self, x):
        return self.conv3d_transpose(x)
class ConvTranspose1(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=2, padding=1, output_padding=1):
        super(ConvTranspose1, self).__init__()
        self.conv3d_transpose1 = nn.ConvTranspose3d(in_channels=in_channels,
                                                   out_channels=out_channels,
                                                   kernel_size=k_size,
                                                   stride=stride,
                                                   padding=padding,
                                                   output_padding=output_padding)

    def forward(self, x):
        return self.conv3d_transpose1(x)

class ConvTranspose2(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=4, stride=2, padding=1, output_padding=1):
        super(ConvTranspose2, self).__init__()
        self.conv3d_transpose2 = nn.ConvTranspose3d(in_channels=in_channels,
                                                   out_channels=out_channels,
                                                   kernel_size=k_size,
                                                   stride=stride,
                                                   padding=padding,
                                                   output_padding=output_padding)

    def forward(self, x):
        return self.conv3d_transpose2(x)




class DecoderBlock(nn.Module):
    def __init__(self, out_channels, model_depth=4):
        super(DecoderBlock, self).__init__()
        self.num_conv_blocks = 2
        self.num_feat_maps = 4
        # user nn.ModuleDict() to store ops
        self.module_dict = nn.ModuleDict()

        for depth in range(model_depth - 2, -1, -1):
            #print(depth)
            feat_map_channels = 2 ** (depth + 1) * self.num_feat_maps
            # print(feat_map_channels * 4)
            self.deconv = ConvTranspose(in_channels=feat_map_channels * 4, out_channels=feat_map_channels * 4)
            self.module_dict["deconv_{}".format(depth)] = self.deconv
            for i in range(self.num_conv_blocks):
                if i == 0:
                    self.conv = ConvBlock(in_channels=feat_map_channels * 6, out_channels=feat_map_channels * 2)
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv
                else:
                    self.conv = ConvBlock(in_channels=feat_map_channels * 2, out_channels=feat_map_channels * 2)
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv
            if depth ==1:
                feat_map_channels = 2 ** (depth + 1) * self.num_feat_maps
                # print(feat_map_channels * 4)
                self.deconv = ConvTranspose1(in_channels=feat_map_channels * 4, out_channels=feat_map_channels * 4)
                self.module_dict["deconv_{}".format(depth)] = self.deconv
                for i in range(self.num_conv_blocks):
                  if i == 0:
                    self.conv = ConvBlock(in_channels=feat_map_channels * 6, out_channels=feat_map_channels * 2)
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv
                  else:
                    self.conv = ConvBlock(in_channels=feat_map_channels * 2, out_channels=feat_map_channels * 2)
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv
            if depth ==0:
                feat_map_channels = 2 ** (depth + 1) * self.num_feat_maps
                #print(feat_map_channels * 4)
                self.deconv = ConvTranspose2(in_channels=feat_map_channels * 4, out_channels=feat_map_channels * 4)
                self.module_dict["deconv_{}".format(depth)] = self.deconv
                for i in range(self.num_conv_blocks):
                  if i == 0:
                    self.conv = ConvBlock(in_channels=feat_map_channels * 6, out_channels=feat_map_channels * 2)
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv
                  else:
                    self.conv = ConvBlock(in_channels=feat_map_channels * 2, out_channels=feat_map_channels * 2)
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv
            if depth == 0:
                self.final_conv = ConvBlock(in_channels=feat_map_channels * 2, out_channels=out_channels)
                self.module_dict["final_conv"] = self.final_conv              

    
    def forward(self, x, down_sampling_features):
        # cn = 0
        #print(self.module_dict)
        for k, op in self.module_dict.items():
            if k.startswith("deconv"):
                #cn+=1
                x = op(x)
                #print(k)
                
                #print(x.shape)
                x = torch.cat((down_sampling_features[int(k[-1])], x), dim=1)
                #print(cn)
                # print(x.shape)
            elif k.startswith("conv"):
                x = op(x)
            else:
                x = op(x)
        return x
'''
if __name__ == "__main__":
    # x has shape of (batch_size, channels, depth, height, width)
    
    sample_image = sys.argv[1]
    sample_image = nib.load(sample_image)
    sample_image = sample_image.get_fdata()
    sample_image = torch.tensor(sample_image)
    input1 = sample_image
    input1 = input1.unsqueeze(0)
    input1 = input1.unsqueeze(0)
    input1 = input1.float()
    #print("The shape of inputs: ", input1.shape)
    
    encoder = EncoderBlock(in_channels=1)
    #print(encoder)
    x_test, h = encoder(input1)
    print(x_test)
    x_test   = torch.load(sys.argv[1])
    h =list(torch.load(sys.argv[2]))
    db = DecoderBlock(out_channels=1)
    x_test = db(x_test, h)
    print(db)
    print("output",x_test.shape)
    '''




class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-5):
        super(DiceLoss, self).__init__()
        # smooth factor
        self.epsilon = epsilon

    def forward(self, targets, logits):
        batch_size = targets.size(0)
        # log_prob = torch.sigmoid(logits)
        logits = logits.view(batch_size, -1).type(torch.FloatTensor)
        targets = targets.view(batch_size, -1).type(torch.FloatTensor)
        intersection = (logits * targets).sum(-1)
        dice_score = 2. * intersection / ((logits + targets).sum(-1) + self.epsilon)
        # dice_score = 1 - dice_score.sum() / batch_size
        return torch.mean(1. - dice_score)


class UnetModel(nn.Module):

    def __init__(self, in_channels, out_channels, model_depth=4, final_activation="sigmoid"):
        super(UnetModel, self).__init__()
        self.encoder = EncoderBlock(in_channels=in_channels, model_depth=model_depth)
        self.decoder = DecoderBlock(out_channels=out_channels, model_depth=model_depth)
        if final_activation == "sigmoid":
            self.sigmoid = nn.Sigmoid()
        else:
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x, downsampling_features = self.encoder(x)
        x = self.decoder(x, downsampling_features)
        x = self.sigmoid(x)
        print("Final output shape: ", x.shape)
        return x



if __name__ == "__main__":
  #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  train = np.load(sys.argv[1])
  #train = torch.from_numpy(train).float()
  #label = np.load(sys.argv[2])
  #label= torch.from_numpy(label).float()
  similar = np.load(sys.argv[2])
  #similar = torch.from_numpy(similar).float()
  UnetModel = UnetModel(in_channels=1, out_channels=1)
  optimizer = torch.optim.Adam(UnetModel.parameters(), lr=1e-1)
single_path = []

#train = train[1]
#print(train.shape)
#similar  = similar[1]
final_loss = 0
epoch = 0
for i,k in zip(train,similar):
    print("Multi  Path Cross")
    total_loss = 0
    image  = i
    similar = k
    input1 =torch.tensor(image)
    input1 = input1.unsqueeze(0)
    input1 = input1.unsqueeze(0)
    input1 = input1.float()
    input2 = torch.tensor(similar)
    input2 = input2.unsqueeze(0)
    input2 = input2.unsqueeze(0)
    input2 = input2.float()
    #print(input2.shape)
    optimizer.zero_grad()
    encoder = EncoderBlock(in_channels=1)
    #encoder2 = EncoderBlock(in_channels=1)
    x_test, h = encoder(input1)
    x_test1, h1 = encoder(input2)
    z = [x_test,x_test1]
    mean = torch.mean(torch.stack(z), dim=0)
    mean_h = [(g + h) / 2 for g, h in zip(h, h1)]
    db = DecoderBlock(out_channels=1)
    d_test = db(mean, mean_h)
    yt = input1
    yp = d_test
    #print(yt.shape,yp.shape)
    #print(dl(yp, yt).item())
    optimizer.zero_grad()
    dl = DiceLoss()
    loss = dl(yp, yt)
    #total_loss += loss.item()
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
    print("loss",total_loss,"Epoch:",epoch)
    final_loss+=total_loss
    epoch+=1
    single_path.append(yp)

    
print(yp.shape)
loss = final_loss/len(train)
print("Mean Dice Loss Single Path",loss)
np.save('Multi_path_images'+'.npy',yp)
