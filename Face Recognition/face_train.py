import os
from PIL import Image
import numpy as np
import cv2
import pickle


x_train=[]
y_label=[]
current_id=0
label_id={}
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

recogniser=cv2.face.LBPHFaceRecognizer_create()


BASE_DIR=os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

for root, dirs, files in os.walk(image_dir):
	for file in files:
		if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
			path = os.path.join(root, file)
			label = os.path.basename(root).replace(" ", "-").lower()
			if not label in label_id:
				label_id[label]=current_id
				current_id+=1
			id_=label_id[label]
			'''x_train.append(path)#Verify this image and turn it into numpy array
			y_label.append(label)#Some Number'''
			pil_image=Image.open(path).convert("L")#Grayscale
			image_array=np.array(pil_image,"uint8")
			faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

			for (x, y, w, h) in faces:
				roi= image_array[y:y + h, x:x + w]
				x_train.append(roi)
				y_label.append(id_)


with open("labels.pickle",'wb') as f:
	pickle.dump(label_id,f)

recogniser.train(x_train,np.array(y_label))
recogniser.save("trainer.yml")

print(label_id)













