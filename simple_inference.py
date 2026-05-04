import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

import clip


img_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"


class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


device = "cuda" if torch.cuda.is_available() else "cpu"

model = MLP(768)
s = torch.load("sac+logos+ava1-l14-linearMSE.pth", map_location=device)
model.load_state_dict(s)
model.to(device)
model.eval()

clip_model, preprocess = clip.load("ViT-L/14", device=device)

pil_image = Image.open(img_path)
image = preprocess(pil_image).unsqueeze(0).to(device)

with torch.no_grad():
    image_features = clip_model.encode_image(image)
    im_emb_arr = normalized(image_features.cpu().detach().numpy())
    prediction = model(torch.from_numpy(im_emb_arr).to(device).float())

print("Aesthetic score predicted by the model:")
print(prediction)
