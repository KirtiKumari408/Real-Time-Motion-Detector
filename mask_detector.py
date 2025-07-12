import winsound
import cv2
import time
import imutils

print("📷 Starting camera...")
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
time.sleep(1)

firstFrame = None
area = 500
last_alert_time = 0  # To avoid constant beeping

print("🎬 Entering main loop...")
while True:
    ret, img = cam.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    print("✅ Frame grabbed")
    text = "Normal"
    img = imutils.resize(img, width=500)
    
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussianImg = cv2.GaussianBlur(grayImg, (21, 21), 0)

    if firstFrame is None:
        print("📸 Capturing first frame")
        firstFrame = gaussianImg
        continue

    imgDiff = cv2.absdiff(firstFrame, gaussianImg)
    threshImg = cv2.threshold(imgDiff, 25, 255, cv2.THRESH_BINARY)[1]
    threshImg = cv2.dilate(threshImg, None, iterations=2)

    cnts = cv2.findContours(threshImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        if cv2.contourArea(c) < area:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Moving Object Detected"

        # 🔊 Beep only when object is detected and not too frequently
        if time.time() - last_alert_time > 2:
            winsound.Beep(1000, 500)  # frequency = 1000 Hz, duration = 500 ms
            last_alert_time = time.time()

    print(text)
    cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow("cameraFeed", img)

    key = cv2.waitKey(10)

    if key == ord("q"):
        print("👋 Q pressed. Exiting now...")
        break


cam.release()
cv2.destroyAllWindows()

