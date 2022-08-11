import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import io
import imgaug.augmenters as iaa


seq = iaa.Sequential(
    [   
        iaa.Resize({"height": 368, "width": 368}),
        iaa.SomeOf((0, 4),
            [
                iaa.Sometimes(0.34, iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(3, 7)), # blur image using local medians with kernel sizes between 2 and 7
                ])),
                iaa.Sometimes(0.3, iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))), # sharpen images 
                # search either for all edges or for directed edges,
                # blend the result with the original image using a blobby mask
                iaa.Sometimes(0.3, iaa.SimplexNoiseAlpha(iaa.OneOf([
                    iaa.EdgeDetect(alpha=(0.5, 1.0)),
                    iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                ]))),
                iaa.Sometimes(0.3, iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5)), # add gaussian noise to images
                iaa.Sometimes(0.45, iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                    iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                ])),
                iaa.Sometimes(0.55, iaa.Add((-10, 10), per_channel=0.5)), # change brightness of images (by -10 to 10 of original value)
                iaa.Sometimes(0.55, iaa.AddToHueAndSaturation((-20, 20))), # change hue and saturation
                # either change the brightness of the whole image (sometimes
                # per channel) or change the brightness of subareas
              
                iaa.Sometimes(0.55, iaa.Multiply((0.5, 1.5), per_channel=0.5)),
                    
                iaa.Sometimes(0.5, iaa.LinearContrast((0.5, 2.0), per_channel=0.5))
            ],
            random_order=True
        )
    ],
    random_order=True
) 

preproc = transforms.Compose([
#     transforms.Resize((368,368)),
    transforms.ToTensor(),
#     normalize,
])
 
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])

iaa_valid = iaa.Resize((368, 368))

def Gaussian(sigma):
    if sigma == 7:
        return np.array([1.23409802e-04, 1.50343915e-03, 6.73794700e-03, 1.11089963e-02,
                     6.73794700e-03, 1.50343915e-03, 1.23409802e-04, 1.50343915e-03,
                     1.83156393e-02, 8.20849985e-02, 1.35335281e-01, 8.20849985e-02,
                     1.83156393e-02, 1.50343915e-03, 6.73794700e-03, 8.20849985e-02,
                     3.67879450e-01, 6.06530666e-01, 3.67879450e-01, 8.20849985e-02,
                     6.73794700e-03, 1.11089963e-02, 1.35335281e-01, 6.06530666e-01,
                     1.00000000e+00, 6.06530666e-01, 1.35335281e-01, 1.11089963e-02,
                     6.73794700e-03, 8.20849985e-02, 3.67879450e-01, 6.06530666e-01,
                     3.67879450e-01, 8.20849985e-02, 6.73794700e-03, 1.50343915e-03,
                     1.83156393e-02, 8.20849985e-02, 1.35335281e-01, 8.20849985e-02,
                     1.83156393e-02, 1.50343915e-03, 1.23409802e-04, 1.50343915e-03,
                     6.73794700e-03, 1.11089963e-02, 6.73794700e-03, 1.50343915e-03,
                     1.23409802e-04]).reshape(7,7)
    elif sigma == n:
        return g_inp
    else:
        raise Exception('Gaussian {} Not Implement'.format(sigma))
        
def DrawGaussian(img, pt, sigma):
    tmpSize = int(np.math.ceil(3 * sigma))
    ul = [int(np.math.floor(pt[0] - tmpSize)), int(np.math.floor(pt[1] - tmpSize))]
    br = [int(np.math.floor(pt[0] + tmpSize)), int(np.math.floor(pt[1] + tmpSize))]

    if ul[0] > img.shape[1] or ul[1] > img.shape[0] or br[0] < 1 or br[1] < 1:
        return img

    size = 2 * tmpSize + 1
    g = Gaussian(size)
    g_x = [max(0, -ul[0]), min(br[0], img.shape[1]) - max(0, ul[0]) + max(0, -ul[0])]
    g_y = [max(0, -ul[1]), min(br[1], img.shape[0]) - max(0, ul[1]) + max(0, -ul[1])]

    img_x = [max(0, ul[0]), min(br[0], img.shape[1])]
    img_y = [max(0, ul[1]), min(br[1], img.shape[0])]

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img

def get_heatmap(points, shape, new_shape=(51,51), joints=6):
    height, width = shape[0], shape[1]
    points[:, 0] = (points[:, 0] / width) * new_shape[1]
    points[:, 1] = (points[:, 1] / height) * new_shape[0]
    points = points.astype(int)
    heatmaps = np.zeros((joints, new_shape[0], new_shape[1]))
    for i in range(joints):
        heatmaps[i] = DrawGaussian(heatmaps[i], (points[i][0], points[i][1]), 1)
    return heatmaps


def transform3D(x, joint_indices):
    x = x.decode("utf-8").split(",")
    x = np.array(x).astype(float).reshape(-1,3)[joint_indices]

    return x.reshape(-1) / 300

def transform2D(x, joint_indices):
    x = x.decode("utf-8").split(",")
    x = np.array(x).astype(float).reshape(-1,2)[joint_indices]
    return x


def pil_decode(data, augment=False):
    img = np.array(Image.open(io.BytesIO(data)))
    if augment:
        img = seq(images=[img])[0] 
    else:
        img = iaa_valid(images=[img])[0]
    img = Image.fromarray(img)

    return img

def decode_train(sample):
    return decode_sample(sample, augment=True)

def decode_valid(sample):
    return decode_sample(sample, augment=False)

def decode_sample(sample, augment=False, indices=[14, 15, 16, 22, 23, 24]): 
    joint_indices = indices
    pose_img = pil_decode(sample["pose_image.png"], augment)
    pose_3dp = transform3D(sample["pose_3dp.csv"], joint_indices)
    width, height = pose_img.size
    points = transform2D(sample["pose_2dp.csv"], joint_indices)
    pose_img = preproc(pose_img)
    heatmaps = get_heatmap(points, (height, width))
    
    return dict(
        image=pose_img,
        pose=pose_3dp,
        heatmaps=heatmaps,
    )