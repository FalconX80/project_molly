'''
import pyttsx3
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id) #changing index changes voices but ony 0 and 1 are working here
engine.say('Hello World')
engine.runAndWait()
'''
from ssml_builder.core import Speech

speech = Speech()
speech.add_text('sample text')
ssml = speech.speak()


while True:
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
                #Speak(label)
                if label=='Sad':
                    Speak('why do you look sad?')
                elif label=='Happy':
                    Speak('good to see you happy.')
                elif label=='Angry':
                    Speak('are you angry?')
                elif label=='Surprize':
                    Speak('did i surprise you?')
                cap.release()
                cv2.destroyAllWindows()