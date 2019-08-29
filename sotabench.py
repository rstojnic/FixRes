from torchbench.image_classification import ImageNet
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import PIL
import urllib.request
import torch
from imnet_evaluate.resnext_wsl import resnext101_32x48d_wsl


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)    

def crop(img, i, j, h, w):
    """Crop the given PIL Image.
        Args:
        img (PIL Image): Image to be cropped.
        i (int): i in (i,j) i.e coordinates of the upper left corner.
        j (int): j in (i,j) i.e coordinates of the upper left corner.
        h (int): Height of the cropped image.
        w (int): Width of the cropped image.
        Returns:
        PIL Image: Cropped image.
        """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    
    return img.crop((j, i, j + w, i + h))


class Resize(transforms.Resize):
    """
    Resize with a ``largest=False'' argument
    allowing to resize to a common largest side without cropping
    """


    def __init__(self, size, largest=False, **kwargs):
        super().__init__(size, **kwargs)
        self.largest = largest

    @staticmethod
    def target_size(w, h, size, largest=False):
        if h < w and largest:
            w, h = size, int(size * h / w)
        else:
            w, h = int(size * w / h), size
        size = (h, w)
        return size

    def __call__(self, img):
        size = self.size
        w, h = img.size
        target_size = self.target_size(w, h, size, self.largest)
        return F.resize(img, target_size, self.interpolation)

    def __repr__(self):
        r = super().__repr__()
        return r[:-1] + ', largest={})'.format(self.largest)
      
# Define the transforms need to convert ImageNet data to expected model input
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
input_transform = transforms.Compose([
    Resize(int((256 / 224) * 320)),
    transforms.CenterCrop(320),
    transforms.ToTensor(),
    normalize,
])

model=resnext101_32x48d_wsl(progress=True) 

urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/FixRes_data/FixRes_Pretrained_Models/ResNext101_32x48d_v2.pth', 'ResNext101_32x48d_v2.pth')
pretrained_dict=torch.load('ResNext101_32x48d_v2.pth',map_location='cpu')['model']

model_dict = model.state_dict()
for k in model_dict.keys():
    if(('module.'+k) in pretrained_dict.keys()):
        model_dict[k]=pretrained_dict.get(('module.'+k))
model.load_state_dict(model_dict)
# Run the benchmark
ImageNet.benchmark(
    model=model,
    paper_model_name='FixResNeXt-101 32x48d',
    paper_arxiv_id='1906.06423',
    input_transform=input_transform,
    batch_size=32,
    num_gpu=1,
    model_description="Official weights from the author's of the paper."
)
