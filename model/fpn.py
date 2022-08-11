from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torch
import torch.nn as nn
from torch.autograd import Variable


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, padding1=1, padding2=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=padding1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=padding2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class HMEncoder(nn.Module):
    def __init__(self, backbone="resnet50", pretrained=True):
        super().__init__()
        self.backbone = resnet_fpn_backbone(backbone_name=backbone, weights="IMAGENET1K_V1")
    
    def forward(self, x):
        outputs = self.backbone(x)
        x56, x28, x14, x7, x4 = outputs['0'], outputs['1'], outputs['2'], outputs['3'], outputs['pool']
        return x56, x28, x14, x7, x4
    

class HMDecoder(nn.Module):
    def __init__(self, feature_size=256, out_channels=6, final_hm=False):
        super().__init__()
        self.final_hm = final_hm
        
        self.x12_deconv1 = nn.ConvTranspose2d(feature_size, feature_size, kernel_size=2, stride=2)
        self.x12_conv1 = DoubleConv(feature_size, feature_size, padding2=1)
        self.x12_deconv2 = nn.ConvTranspose2d(feature_size, feature_size, kernel_size=2, stride=2)
        self.x12_conv2 = DoubleConv(feature_size, feature_size, padding2=1, padding1=0)
        
        self.x23_deconv1 = nn.ConvTranspose2d(feature_size, feature_size, kernel_size=2, stride=2)
        self.x23_conv1 = DoubleConv(feature_size, feature_size, padding2=1)
        
        self.conv92 = DoubleConv(feature_size, feature_size)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        if self.final_hm:
            self.conv = DoubleConv(feature_size, feature_size, padding1=2)
            self.conv2 = DoubleConv(feature_size, feature_size, padding1=2)
            self.conv3 = DoubleConv(feature_size, out_channels, padding1=1)
        else:
            self.conv = DoubleConv(feature_size*4, feature_size, padding1=2)
            self.conv2 = DoubleConv(feature_size, feature_size//2, padding1=2)
            self.conv3 = DoubleConv(feature_size//2, out_channels, padding1=1)
        
    def forward(self, x12, x23=None, x46=None, x92=None):
        x12 = self.x12_deconv1(x12)
        x12 = self.x12_conv1(x12)
        x12 = self.x12_deconv2(x12)
        x12 = self.x12_conv2(x12)
        if not self.final_hm:
            x23 = self.x23_deconv1(x23)
            x23 = self.x23_conv1(x23)

            x92 = self.conv92(x92)
            x92 = self.pool(x92)

            x = torch.cat([x12, x23, x46, x92], 1)
        else:
            x = x12
        x = self.conv(x)
        x = self.conv2(x)
        x = self.conv3(x)
       

        return x
    
    
class Regressor(nn.Module):
    def __init__(self, joints=6):
        super().__init__()
        self.conv1 = DoubleConv(joints,64)
        self.conv2 = DoubleConv(64,128)
        self.conv3 = DoubleConv(128, 128, padding2=0)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.l_relu = nn.LeakyReLU(0.2)
        
        self.fc1 = nn.Linear(15488, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 50)
        
        self.FC_mean  = nn.Linear(50, 50)
        self.FC_var   = nn.Linear (50, 50)
        
        self.pose_fc1 = nn.Linear(50, 32)
        self.pose_fc2 = nn.Linear(32, 32)
        self.pose_fc3 = nn.Linear(32, joints * 3)
        
        self.hm_fc1 = nn.Linear(50, 512)
        self.hm_fc2 = nn.Linear(512, 2048)
        self.hm_fc3 = nn.Linear(2048, 18432)
        
        self.hm_decoder = HMDecoder(128, out_channels=joints, final_hm=True)
    
    def reparameterization(self, mean, var):
        std = var.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
#         eps = torch.rand((10, 64, 50)).cuda()        # sampling epsilon       
        z = mean + std*eps                          # reparameterization trick
#         z = (eps * std.repeat(10,1,1)) + mean.repeat(10,1,1)
        return z
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.l_relu(self.fc1(x))
        x = self.l_relu(self.fc2(x))
        x = self.l_relu(self.fc3(x))
        
        mean = self.FC_mean(x)
        var = self.FC_var(x)
        z = self.reparameterization(mean, torch.exp(0.5 * var))
        poses = []
#         for i in range(z.shape[0]):
        pose = self.l_relu(self.pose_fc1(z))
        pose = self.l_relu(self.pose_fc2(pose))
        pose = self.l_relu(self.pose_fc3(pose))

#         rotation = self.l_relu(self.rotation_fc1(x))
#         rotation = self.l_relu(self.rotation_fc2(rotation))
#         rotation = self.rotation_fc3(rotation)

        hm = self.l_relu(self.hm_fc1(z))
        hm = self.l_relu(self.hm_fc2(hm))
        hm = self.l_relu(self.hm_fc3(hm))
        hm = hm.view(hm.size(0),128,12,12)
        hm = self.hm_decoder(hm)

        return pose, hm, mean, var
        
class SelfPose(nn.Module):
    def __init__(self, backbone="resnet50", pretrained=True, feature_size=2048, joints=6):
        super().__init__()
        self.encoder = HMEncoder(backbone)
        self.decoder = HMDecoder(256, joints)
        self.regressor = Regressor(joints)
    
    def forward(self, x):
        x92, x46, x23, x12, x4 = self.encoder(x)

        hm = self.decoder(x12, x23, x46, x92)
        pose, pred_hm, mean, var = self.regressor(hm)
        return pose, hm, pred_hm, mean, var
    