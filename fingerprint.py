import cv2
import torch
from torch import nn, optim
from torchvision import transforms, datasets, models
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import numpy as np
import mysql_code as mc

students=[]

def main():
    class_names = ['142000', '142001', '142003', '142004', '142005', '142006', '142007', '142008', '142010', '142011']
    model = models.densenet121(pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, 10)
    model.load_state_dict(torch.load('fingerprint_DenesNet-121.pt'))
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    image = Image.open(file_path)
    prediction_transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                               transforms.ToTensor()])
    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = prediction_transform(image)[:3, :, :].unsqueeze(0)
    model = model.cpu()
    model.eval()
    idx = torch.argmax(model(image))
    students=np.array(class_names[idx])
    print(students)
    img=cv2.imread(file_path)
    cv2.imshow("fingerprint",img)
    cv2.waitKey()
    return students

student=[]

student.append(main())

mc.main(student)




