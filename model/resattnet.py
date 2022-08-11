from pytorchcv.model_provider import get_model
import torch
import torch.nn as nn
from torch.autograd import Variable


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, padding=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class HMEncoder(nn.Module):
    def __init__(self, backbone="resnet50", pretrained=True):
        super().__init__()
        model = get_model("resattnet56", pretrained=False)
        path = """https://github.com/phamquiluan/ResidualAttentionNetwork/releases/download/v0.1.0/resattnet56.pth"""
        state = torch.hub.load_state_dict_from_url(path)
        model.load_state_dict(state["state_dict"])
        self.model = nn.Sequential(*list(model.features.children())[:-1])
    
    def forward(self, x):
        x = self.model(x)
        return x
    

class HMDecoder(nn.Module):
    def __init__(self, feature_size=2048, out_channels=6):
        super().__init__()
        
        self.deconv1 = nn.ConvTranspose2d(feature_size, feature_size // 2, kernel_size=3, stride=2)
        self.conv1 = DoubleConv(feature_size // 2, feature_size // 2)
        self.deconv2 = nn.ConvTranspose2d(feature_size // 2, feature_size // 4, kernel_size=3, stride=2)
        self.conv2 = DoubleConv(feature_size // 4, feature_size // 8)
        self.conv3 = DoubleConv(feature_size // 8, out_channels)
        
        
    def forward(self, x):
        x = self.deconv1(x)
        x = self.conv1(x)
        x = self.deconv2(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x
    
    
class Regressor(nn.Module):
    def __init__(self, joints=6):
        super().__init__()
        self.conv1 = DoubleConv(joints,64)
        self.conv2 = DoubleConv(64,128)
        self.conv3 = DoubleConv(128, 128, padding=0)
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
        
        self.hm_decoder = HMDecoder(128, out_channels=joints)
    
    def reparameterization(self, mean, var):
        std = var.mul(0.5).exp_()
#         eps = Variable(std.data.new(std.size()).normal_())
#         mean = mean.repeat(21,1,1)
        eps = torch.rand((std.shape[0], std.shape[-1])).cuda()
        z = (eps * std) + mean
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

        pose = self.l_relu(self.pose_fc1(z))
        pose = self.l_relu(self.pose_fc2(pose))
        pose = self.l_relu(self.pose_fc3(pose))


        hm = self.l_relu(self.hm_fc1(z))
        hm = self.l_relu(self.hm_fc2(hm))
        hm = self.l_relu(self.hm_fc3(hm))
        hm = hm.view(hm.size(0),128,12,12)
        hm = self.hm_decoder(hm)

    #             break
        return pose, hm, mean, var
        
class SelfPose(nn.Module):
    def __init__(self, backbone="resnet50", pretrained=True, feature_size=2048, joints=6):
        super().__init__()
        self.encoder = HMEncoder(backbone)
        self.decoder = HMDecoder(feature_size, joints)
        self.regressor = Regressor(joints)
    
    def forward(self, x):
        x = self.encoder(x)
        hm = self.decoder(x)
        pose, pred_hm, mean, var = self.regressor(hm)
        
        return pose, hm, pred_hm, mean, var
    
    
if __name__ == "__main__":
    model = SelfPose()
    im = torch.zeros((1,3,384,384))
    model(im)
    
    
    
    
        
    