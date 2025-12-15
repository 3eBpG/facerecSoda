# подключаем библиотеку компьютерного зрения
import cv2
# библиотека для вызова системных функций
import os
# для обучения нейросетей
import numpy as np
# встроенная библиотека для работы с изображениями
from PIL import Image 
# получаем путь к этому скрипту
path = os.path.dirname(os.path.abspath(__file__))
#print(path)
# создаём новый распознаватель лиц
recognizer = cv2.face.LBPHFaceRecognizer_create(1,8,8,8,123)
# указываем, что мы будем искать лица по примитивам Хаара
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# путь к датасету с фотографиями пользователей
dataPath = path+r'/data_set'
#print(dataPath)
font = cv2.FONT_HERSHEY_SIMPLEX
# получаем картинки и подписи из датасета
def get_images_and_labels(datapath):
     # получаем путь к картинкам
     image_paths = [os.path.join(datapath, f) for f in os.listdir(datapath)]
     # списки картинок и подписей на старте пустые
     images = []
     labels = []

     # перебираем все картинки в датасете 
     for image_path in image_paths:
         # читаем картинку и сразу переводим в ч/б
         image_pil = Image.open(image_path).convert('L')
         # переводим картинку в numpy-массив
         image = np.array(image_pil, 'uint8')
         # получаем id пользователя из имени файла
         nbr =int((os.path.split(image_path)[1].split(".")[0].split("_"))[0])
         #print(nbr)
         #return 1
         # определяем лицо на картинке
         faces = faceCascade.detectMultiScale(image)
         # если лицо найдено
         
         for (x, y, w, h) in faces:
             # добавляем его к списку картинок 
             images.append(image[y: y + h, x: x + w])
             # добавляем id пользователя в список подписей
             labels.append(nbr)
             # выводим текущую картинку на экран
             im=image[y: y + h, x: x + w]
             print(nbr)
             #cv2.putText(im,str(nbr), (x,y+h),font, 1.1, (0,255,0))
             cv2.imshow("Adding faces to traning set...", im)
             # делаем паузу
             cv2.waitKey(1000)
     # возвращаем список картинок и подписей
     return images, labels

# получаем список картинок и подписей
images, labels = get_images_and_labels(dataPath)
# обучаем модель распознавания на наших картинках и учим сопоставлять её лица и подписи к ним
recognizer.train(images, np.array(labels))
# сохраняем модель
recognizer.save(path+r'/trainer/trainer.yml')
print(path+r'/trainer/trainer.yml')
# удаляем из памяти все созданные окнаы
#cv2.destroyAllWindows()
