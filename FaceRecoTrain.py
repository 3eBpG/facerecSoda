import face_recognition
import cv2
import numpy as np
import os
import pickle
# Create arrays of known face encodings and their names and save to file to use in the next programm
known_face_encodings = []
all_face_encodings={}
known_face_names = []
path = os.path.dirname(os.path.abspath(__file__))
datapath = path+r'/data_set3'
image_paths = [os.path.join(datapath, f) for f in os.listdir(datapath)]
for image_path in image_paths:
        # читаем картинку и сразу переводим в rgb
        img = cv2.imread(image_path)[:,:,::-1]
        img= cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
        print(image_path)
        x_image = face_recognition.load_image_file(image_path)
        #print(x_image)
        #x_face_encoding = face_recognition.face_encodings(img)[0]
        x_temp_encd = face_recognition.face_encodings(img)
        if len(x_temp_encd)>0:
            x_face_encoding=x_temp_encd[0]
        else:
            print("dunno looks like bro cant find a face")
        nbr =((os.path.split(image_path)[1].split(".")[0].split("_"))[1])
        all_face_encodings[nbr]=x_face_encoding
        known_face_encodings.append(x_face_encoding)
        known_face_names.append(nbr)
        #cv2.imshow("Adding faces to traning set...", im)

#print(all_face_encodings)

with open(path+r'\base\knownFace.dat','wb') as f:
    pickle.dump(all_face_encodings, f)

                        
