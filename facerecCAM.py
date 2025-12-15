# подключаем библиотеку компьютерного зрения
import cv2
# библиотека для вызова системных функций
import os
def saypay(sale):
    #скидывает коду для оплаты процент скидки sale%
    pass
def gotodatabase(idd):
    #обращение к датабазе
    #dunnohow p=getperson(database)
##    if p.rating>=4.5:
##        #def saypay(100)
##        print("бесплатный напиток для ",p.name)
##    elif p.rating>=4:
##        #def saypay(50)
##        print("скидка 50% ",p.name)
##    else:
##        #def saypay(0)
##        print("скидка 0% ",p.name)
    print("типо обратился к бд для чела с id= ",idd)
    pass
# получаем путь к этому скрипту
path = os.path.dirname(os.path.abspath(__file__))
# создаём новый распознаватель лиц
recognizer = cv2.face.LBPHFaceRecognizer_create(1,10,8,8,200)
# добавляем в него модель, которую мы обучили на прошлых этапах
recognizer.read(path+r'/trainer/trainer.yml')
# указываем, что мы будем искать лица по примитивам Хаара
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# получаем доступ к камере
cam = cv2.VideoCapture(0)

# настраиваем шрифт для вывода подписей
font = cv2.FONT_HERSHEY_SIMPLEX

# запускаем цикл
j=1
i=1
accuracy=[]
while True:
    #print("gotcha")
    # получаем видеопоток
    ret, im =cam.read()
    
    if j:
        print("i should have started cam")
        j=0
    #print("gotcha2")
    # переводим его в ч/б
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    # определяем лица на видео
    faces=faceCascade.detectMultiScale(gray,scaleFactor=1.2, minNeighbors=6, minSize=(50, 50), flags=cv2.CASCADE_SCALE_IMAGE)
    # перебираем все найденные лица
    if i:
        print("i should have been started")
    for(x,y,w,h) in faces:
        i=0
        #print("face")
        # получаем id пользователя
        nbr_predicted,coord = recognizer.predict(gray[y:y+h,x:x+w])
        # рисуем прямоугольник вокруг л≈ица
        cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)
        # если мы знаем id пользователя
        accuracy.append(nbr_predicted)
        if accuracy.count(nbr_predicted)>10:
            gotodatabase(nbr_predicted)
            accuracy.clear()
        if(nbr_predicted==8):
             # подставляем вместо него имя человека
            nbr_predicted='3ebp'
        elif(nbr_predicted==13):
            nbr_predicted='Alisa <3'
        elif(nbr_predicted==15):
            nbr_predicted='vanya'
        elif(nbr_predicted==17):
            nbr_predicted='GIGAkiruha'
        elif(nbr_predicted==20):
            nbr_predicted='SIGMA POKIDOV'
        elif(nbr_predicted==21):
            nbr_predicted='GIGACHAD PAHSA SELIN'
        elif(nbr_predicted==25):
            nbr_predicted='vika'
        # добавляем текст к рамке
        cv2.putText(im,str(nbr_predicted), (x,y+h),font, 1.1, (0,255,0))
        # выводим окно с изображением с камеры
        cv2.imshow('Face recognition',im)
        # делаем паузу
        cv2.waitKey(10)
