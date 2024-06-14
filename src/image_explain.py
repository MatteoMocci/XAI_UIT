'''
08-08-23
Questo programma genera delle spiegazioni Shap utilizzando una ResNet pretrainata su ImageNet da PyTorch
'''
import json
import torch
from torchvision import models, transforms
from PIL import Image as PilImage

from omnixai.preprocessing.image import Resize
from omnixai.data.image import Image
from omnixai.explainers.vision import VisionExplainer
from omnixai.visualization.dashboard import Dashboard
import time

def get_target_layer(model, model_name):
    if model_name == 'resnet50':
        return model.layer4[-1]
    elif model_name == 'alexnet':
        return model.features[-1]
    else:
        raise ValueError("Model not supported for Grad-CAM.")


def image_explain_script(input_pic = None, socket = None, model_name='resnet50'):
    if input_pic is None:
        input_pic = Image(PilImage.open('camera.jpg').convert('RGB'))
    else:
        input_pic = Image(PilImage.open(input_pic).convert('RGB'))  
    # Load the test image
    img = Resize((256, 256)).transform(input_pic)

    # Load the class names
    with open('imagenet_class_index.json', 'r') as read_file:
        class_idx = json.load(read_file)
        idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # The preprocessing function
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    preprocess = lambda ims: torch.stack([transform(im.to_pil()) for im in ims]).to(device)
    # A ResNet model to explain
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True).to(device)
    elif model_name == 'alexnet':
        model = models.alexnet(pretrained=True).to(device)
    else:
        print("Errore!")
    # The postprocessing function
    postprocess = lambda logits: torch.nn.functional.softmax(logits, dim=1)

    explainer = VisionExplainer(
        explainers=["lime", "ig", "gradcam"],
        mode="classification",
        model=model,
        preprocess=preprocess,
        postprocess=postprocess,
        params={
            "gradcam": {"target_layer": get_target_layer(model, model_name)},
        }
    )

    # Generate explanations
    local_explanations = explainer.explain(img)

    dashboard = Dashboard(
        instances=img,
        local_explanations=local_explanations,
        class_names=idx2label,
    )

    if socket is not None:
        time.sleep(5)
        socket.emit('dashboard_status', {'running': True})
    
    dashboard.show()

if __name__ == '__main__':
    image_explain_script()