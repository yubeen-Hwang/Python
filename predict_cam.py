import numpy as np
import cv2
from deep_convnet import DeepConvNet
from common.image import img2gray, resize, find_bounding_rects, visualize_rects
from common.functions import softmax



# 기본 카메라 장치 열기
cap = cv2.VideoCapture(0)

network = DeepConvNet()
network.load_params("params.pkl")

while cap.isOpened():
    # 카메라로부터 프레임을 정상적으로 받아오면 ret에는 True,
    ret, img = cap.read()

    if ret:
        img_Gray = img2gray(img)
        img_Gray = img_Gray[0:img_Gray.shape[0]-10, 15:img_Gray.shape[1]-25]
        img_Gray = cv2.GaussianBlur(img_Gray, (5, 5), 0)
        _, threshold = cv2.threshold(img_Gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        edges = cv2.Canny(img_Gray, 100, 200)

        def resize_with_padding(image, size=(28, 28)):
            h, w = image.shape[:2]
            scale = min(size[0] / h, size[1] / w)
            new_h, new_w = int(h * scale), int(w * scale)
            resized = cv2.resize(image, (new_w, new_h))
            pad_h = (size[0] - new_h) // 2
            pad_w = (size[1] - new_w) // 2
            padded = cv2.copyMakeBorder(resized, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=255)
            return padded
    #이미지 이진화 및 전처리
    
    
        rects = find_bounding_rects(img_Gray, min_size=(10, 10)) 
    #이미지에서 숫자 영역(contour)들을 찾아내기
        #img = visualize_rects(img, rects) # 숫자 영역에 사각형 그리기

    

    for rect in rects: # 각각의 숫자 영역에서 숫자를 추출해서 예측 결과를 출력
        x, y, w, h = rect
        rect_img = img_Gray[y:y+h, x:x+w] # 숫자 부분 잘라내기
        resized_img = resize(rect_img, dsize=(28, 28)).reshape(1,1,28,28)
        
        
        prediction = softmax(network.predict(resized_img / 255.0))[0]
        index = np.argmax(prediction)
        accuracy = prediction[index]

        if accuracy > 0.8:
            visualize_rects(img, rect)
            cv2.putText(img, f'{index} ({round(accuracy, 2)})', (x, y),
                            cv2.FONT_HERSHEY_PLAIN, 3, color=(255, 0, 0), thickness=2)
     
            



        cv2.imshow('Webcam', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # 일정 시간 기다린 후 다음 프레임 처리.
    # 만약 ESC 키를 누르면 while 루프 종료

# 사용한 자원 해제
cap.release()
cv2.destroyAllWindows()

