
import face_recognition
import cv2
import numpy as np
import os
import pickle
import psycopg2  # Добавлен импорт для PostgreSQL

# Функция для проверки подключения к PostgreSQL
def check_postgresql_connection():
    """
    Проверяет подключение к базе данных PostgreSQL
    Возвращает True если подключение успешно, False в противном случае
    """
    try:
        # Настройки подключения (замените на свои)
        connection = psycopg2.connect(
            host="localhost",          # или IP адрес сервера
            database="yir_bd",  # имя базы данных
            user="postgres",          # имя пользователя
            password="Fktif2006",  # пароль
            port=5432                 # порт PostgreSQL
        )
        
        # Создаем курсор для выполнения запросов
        cursor = connection.cursor()
        
        # Выполняем простой запрос для проверки подключения
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        
        # Закрываем курсор и соединение
        cursor.close()
        connection.close()
        
        print("✓ Успешное подключение к PostgreSQL")
        return True
        
    except psycopg2.OperationalError as e:
        print(f"✗ Ошибка подключения к PostgreSQL: {e}")
        print("  Проверьте настройки подключения в коде")
        return False
    except Exception as e:
        print(f"✗ Неизвестная ошибка при подключении к PostgreSQL: {e}")
        return False

# Проверяем подключение к базе данных перед началом работы
print("=" * 50)
print("Проверка подключения к базе данных PostgreSQL...")
db_connected = check_postgresql_connection()
print("=" * 50)

# Get a reference to webcam #0 (the default one)

# Create arrays of known face encodings and their names
known_face_encodings = []
known_face_names = []
path = os.path.dirname(os.path.abspath(__file__))

print("xd")
##f=open(path+r'\base\knownFace.txt','w')
##f.write(known_face_encodings[0])

with open(path+r'\base\knownFace.dat', 'rb') as f:
	all_face_encodings = pickle.load(f)

# Grab the list of names and the list of encodings
known_face_names = list(all_face_encodings.keys())
known_face_encodings = np.array(list(all_face_encodings.values()))

video_capture = cv2.VideoCapture(0)
# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Only process every other frame of video to save time
    if process_this_frame:
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        rgb_small_frame = cv2.cvtColor(rgb_small_frame , cv2.COLOR_BGR2RGB)
        
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    
    # Добавляем информацию о статусе подключения к БД (опционально)
    font = cv2.FONT_HERSHEY_SIMPLEX
    db_status = "DB: Connected" if db_connected else "DB: Disconnected"
    cv2.putText(frame, db_status, (10, 30), font, 0.7, (0, 255, 0) if db_connected else (0, 0, 255), 2)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
