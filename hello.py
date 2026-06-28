import cv2
import numpy as np
import requests

xml_model_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
xml_path = "haarcascade_frontalface_default.xml"

def download_xml(url, path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(path, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded XML model to {path} successfully.")
    else:
        raise Exception(f"Failed to download XML model. Status code: {response.status_code}")

download_xml(xml_model_url, xml_path)
cascade = cv2.CascadeClassifier(xml_path)

if cascade.empty():
    raise Exception("Failed to Load the Haarcascade XML model.")

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    objects = cascade.detectMultiScale(gray, scaLeFactors=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in object:
        cv2.imshow(frame, (x, y), (x + w, y + h), (225, 225, 0), 2)

    cv2.imshow("Face Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()