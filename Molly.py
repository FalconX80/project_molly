from Brain.AIBrain import ReplyBrain
from Body.Listen import MicExecution

import cv2
from simple_facerec import SimpleFacerec
from Body.Speak import Speak
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import numpy as np

sfr = SimpleFacerec()
sfr.load_encoding_images("images/")
print("Please Wait. Starting...")




def MainExecution():

    Speak("Hello Sir")
    Speak("I am Molly, Ready to assist you Sir.")
    while True:
        Data = MicExecution()
        Data = str(Data)
        if len(Data)<2:
            pass
        elif 'Hi' in Data or 'Hello' in Data or 'How are you' in Data:
            print(Data)
            # Load Camera
            cap = cv2.VideoCapture(0)

            while True:
                ret, frame = cap.read()
                namelist=[]
                # Detect Faces
                face_locations, face_names = sfr.detect_known_faces(frame)
                for face_loc, name in zip(face_locations, face_names):
                    y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

                    cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
                    namelist.append(name)
                #print(namelist)
                cv2.imshow("Frame", frame)
                try:
                    if len(namelist)>=1:
                        if namelist[0]!='Unknown':
                            Speak(f'hello {namelist[0]} nice to meet you')
                            break
                        else:
                            Speak('hello, how are you?')
                            break
                except Exception as e:
                    pass

            cap.release()
            cv2.destroyAllWindows()
            
            face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
            classifier =load_model(r'model.h5')

            emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
            cap = cv2.VideoCapture(0)
            while True:
                try:
                    _, frame = cap.read()
                    labels = []
                    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                    faces = face_classifier.detectMultiScale(gray)
                    
                    for (x,y,w,h) in faces:
                        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
                        roi_gray = gray[y:y+h,x:x+w]
                        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)



                        if np.sum([roi_gray])!=0:
                            roi = roi_gray.astype('float')/255.0
                            roi = img_to_array(roi)
                            roi = np.expand_dims(roi,axis=0)

                            prediction = classifier.predict(roi)[0]
                            label=emotion_labels[prediction.argmax()]
                            label_position = (x,y)
                            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                            
                        else:
                            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                        
                    cv2.imshow('Emotion Detector',frame)
                    try:
                        if label=='Sad':
                            Speak('why do you look sad?')
                            cap.release()
                            cv2.destroyAllWindows()
                        elif label=='Happy':
                            Speak('good to see you happy.')
                            cap.release()
                            cv2.destroyAllWindows()
                        elif label=='Angry':
                            Speak('are you angry?')
                            cap.release()
                            cv2.destroyAllWindows()
                        elif label=='Surprise':
                            Speak('did i surprise you?')  
                            cap.release()
                            cv2.destroyAllWindows()
                    except Exception as e:
                        pass
                
                        
                    cap.release()
                    cv2.destroyAllWindows()

                except Exception as e:
                    break

        else:
            Reply = ReplyBrain(Data)
            Speak(Reply)
MainExecution()