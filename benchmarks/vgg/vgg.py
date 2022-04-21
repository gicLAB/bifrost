


import urllib
from PIL import Image




url, filename = ("https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)



from torchvision import transforms
from torchvision.models import vgg11

# Instansiate torch model
vgg11_torch = vgg11(pretrained = True)

# Get input
input_image = Image.open("dog.jpg")
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
        ),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) 
