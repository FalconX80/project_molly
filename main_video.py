import cv2
from simple_facerec import SimpleFacerec
from Body.Speak import Speak
# Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

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
                Speak('hello how are you?')
    except Exception as e:
        pass
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
