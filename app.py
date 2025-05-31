import cv2
import time
import mediapipe as mp

# 0: varsayılan kamera
cap = cv2.VideoCapture(0)
mpHand = mp.solutions.hands
hands = mpHand.Hands()
mpDraw = mp.solutions.drawing_utils
pTime = 0
cTime = 0
# sonsuz döngü – her kareyi okuyup ekrana basar
while True:
    # kameradan bir kare alır – success: okuma başarılı mı, img: alınan görüntü
    success, img = cap.read() #• kameradan bir kare okur
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # mediapipe rgb ile çalışıyor
    
    results = hands.process(imgRGB) # rgb görüntüde el algılama 
    print(results.multi_hand_landmarks)
    
    # eğer algılanan bir kordinat varsa
    if results.multi_hand_landmarks:
        # her algılanan el için
        for handLms in results.multi_hand_landmarks:
            #elin tüm eklemlerini çizer
            mpDraw.draw_landmarks(img, handLms, mpHand.HAND_CONNECTIONS)
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape # görüntü boyutu
                cx, cy = int(lm.x*w), int(lm.y*h) # normalleştirilmiş kordinatları pixel cinsine çevirme
                
                if id == 12:
                    cv2.circle(img, (cx,cy), 9, (255,0,255), cv2.FILLED)
                    
    # FPS Hesaplama
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    
    cv2.putText(img, "FPS : "+str(int(fps)), (10,75), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,0), 5)
    cv2.imshow("img", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows() 
