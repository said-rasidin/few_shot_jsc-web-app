import torch
from torchvision import transforms
import io
import numpy as np
from PIL import Image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SUPPORT_IMG = torch.load("app/support_images/support_img.pt")
SUPPORT_LABEL = torch.load("app/support_images/support_label.pt")
INDEX_LABEL = {0: 'Clean Water Network',
                1: 'Communication Network',
                2: 'Flood',
                3: 'Garbage',
                4: 'Gutter Cover',
                5: 'Illegal Parking',
                6: 'Layout and Building',
                7: 'Park',
                8: 'Road',
                9: 'Sidewalk',
                10: 'Tree',
                11: 'Vandalism',
                12: 'Waterway'}

def predict_one_img(model, image, 
                    API =False,
                    support_img=SUPPORT_IMG, 
                    support_label=SUPPORT_LABEL, 
                    index_label=INDEX_LABEL):
    """
    support and query image going forward to model and prediction result
    return -> array value for all classes;
           -> max value
           -> class index
    """
    if API:
        img = Image.open(io.BytesIO(image))
    else:
        img = Image.open(image)
    model.eval()
    model.to(device)
    imsize = (84, 84)
    with torch.no_grad():
        loader = transforms.Compose([transforms.Resize(imsize), 
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                        std = [0.229, 0.224, 0.225])])
        image = loader(img)
        image = image.unsqueeze(0)

        model.process_support_set(support_img.to(device), support_label.to(device))
        pred = model(image.to(device))
        pred = np.array(pred.squeeze(0))
        similarity = []
        for i in range(len(pred)):
            #change euclidean dist to similarity value
            similarity.append(1 - np.abs(pred[i]) / np.max(np.abs(pred)).item())
        similarity_perct = [round(x*100,2) for x in similarity]
    return similarity_perct, np.max(similarity_perct), INDEX_LABEL[np.argmax(similarity_perct)]



def load_model(fpath):
    """
    load model weights after training
    """
    check = torch.load(fpath, map_location=torch.device(device) )
    model = check['model']
    model.load_state_dict(check['state_dict'])
    return model