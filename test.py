from model.fpn import SelfPose
import torch

if __name__ == "__main__":
    model = SelfPose().cuda()
    im = torch.zeros((1,3,368,368)).cuda()
    model(im)