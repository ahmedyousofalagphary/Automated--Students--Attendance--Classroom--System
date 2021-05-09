import cv2
import pickle
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
import mysql_code as mc
import tkinter as tk
from tkinter import filedialog

class_name=['']
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
model = models.alexnet(pretrained=True)
model.classifier[6] = nn.Linear(4096, 100)
model.load_state_dict(torch.load('facealex.pt'))

prediction_transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                               transforms.ToTensor()])

studenid=[]

cap = cv2.VideoCapture(0)
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()

scale_percent = 30
frame=cv2.imread(file_path)
width = int(frame.shape[1] * scale_percent / 100)
height = int(frame.shape[0] * scale_percent / 100)
frame = cv2.resize(frame, (width, height))

gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=3)
for (x, y, w, h) in faces:

    roi_gray = gray[y:y+h, x:x+w]
    roi_color=(frame[y:y+h, x:x+w])

    im_pil = Image.fromarray(roi_color)
    image = prediction_transform(im_pil)[:3, :, :].unsqueeze(0)
    model = model.cpu()
    model.eval()
    idx = torch.argmax(model(image))
    font = cv2.FONT_HERSHEY_SIMPLEX
    name = class_name[idx]
    color = (0, 0, 255)
    stroke = 2
    print(name)
    cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)


    color = (0, 0, 255) #BGR 0-255
    stroke = 2
    end_cord_x = x + w
    end_cord_y = y + h
    cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
    studenid.append(name)


cv2.imshow('frame',frame)
cv2.waitKey(0)

mc.main(studenid)

