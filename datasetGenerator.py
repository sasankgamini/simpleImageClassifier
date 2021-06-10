import cv2

capture=cv2.VideoCapture(0)
n=1
activation=False
while n<=300:
    _, frame = capture.read()
    cv2.imshow('video',frame)

    if cv2.waitKey(3) == ord('s'):
        print('activated')
        activation=True
    if activation == True:
        frame=cv2.resize(frame,(100,100))
        cv2.imwrite('Something/something'+str(n)+'.png',frame)
        n+=1


capture.release()
cv2.destroyAllWindows()
