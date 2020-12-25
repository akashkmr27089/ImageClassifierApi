from torchvision import transforms, datasets, models
from PIL import Image
from torchsummary import summary
from ModelDict import Vgg16Labels

# Image transformations
imageResize = transforms.Compose([
        transforms.CenterCrop(size=224),  # Image net standards
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


# Final Execution 
img = Image.open('./image2.jpg')

transfromed_img = imageResize(img)
final_image = transfromed_img.resize(1,3,224,224)
model = models.vgg16(pretrained=True)
Predicted = model(final_image)

print("Final Predicted Value :: {}".format(Vgg16Labels[int(Predicted.argmax())]))