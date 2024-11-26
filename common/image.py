import cv2
from PIL import Image
import numpy as np


def img_show(img): # 이미지 보여주기 함수
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


def img2gray(img): # 이미지를 흑백으로 변환하는 함수
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV) # 이진화 처리

    return binary_img


def resize(img, dsize): # 이미지 크기를 조정하는 함수
    return cv2.resize(img, dsize, interpolation=cv2.INTER_AREA)


def find_bounding_rects(img, min_size=(50, 50)): # 이미지의 contour 를 찾아서 bounding box 를 추출하는 함수
    '''
    find_bounding_rects() 함수는 입력받은 이미지에서 contour를 찾아서 bounding box를 추출하는 함수입니다.

    함수 내부에서는 먼저 cv2.GaussianBlur() 함수를 사용해서 이미지를 부드럽게 만들어 줍니다.
    이렇게 하면 contour 추출 시 노이즈가 줄어들기 때문에 더욱 정확한 contour를 추출할 수 있습니다.

    그 다음 cv2.findContours() 함수를 사용해서 이미지에서 contour를 추출합니다.
    이때, RETR_CCOMP 옵션을 사용하면 모든 contour를 추출하는 대신,
    contour의 외곽선과 내부 구멍의 contour만 추출할 수 있습니다.
    이렇게 하면 이미지 전체를 다루기가 어려운 경우 더욱 쉽게 contour를 추출할 수 있습니다.

    그 다음 가장 큰 contour를 선택합니다.
    이때 cv2.contourArea() 함수를 이용해 각 contour의 면적을 계산하고, 그 중 가장 큰 것을 선택합니다.
    이 작업을 통해 컨투어를 감싸는 최소한의 직사각형 영역을 찾을 수 있습니다.

    그 다음 cv2.drawContours() 함수를 이용해 선택된 contour로 이미지를 다시 그립니다.
    이렇게 하면 전체 이미지를 다루기보다, 관심 영역만 추출할 수 있습니다.

    마지막으로 cv2.RETR_EXTERNAL 옵션과 cv2.boundingRect() 함수를 이용해 bounding box를 추출합니다.
    이렇게 추출된 bounding box 중에서, min_size로 지정한 최소 크기보다 작은 것들은 제외하고
    나머지만 반환합니다.

    따라서 이 함수는 이미지 전체를 다루지 않고도 관심 영역만 추출할 수 있어,
    이미지 처리 과정을 더욱 빠르고 정확하게 할 수 있습니다.
    '''

    img = cv2.GaussianBlur(img, (5, 5), 0) # 가우시안 필터로 이미지를 부드럽게 만들어줌

    contours, _ = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE) # 윤각 추출

    contour_sizes = [(cv2.contourArea(contour), contour)  # 가장 큰 contour 을 선택
                     for contour in contours]
    max_contour = max(contour_sizes, key=lambda x: x[0])[1]

    cv2.drawContours(img, max_contour, -1, (255, 255, 255)) # 선택한 contour 로 이미지 다시 그리기

    # contour 다시 추출
    contours, _ = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # bounding box 추출하고, 사이즈가 작은 것들은 제외함
    rects = [cv2.boundingRect(contour) for contour in contours]
    rects = sorted(
        rect for rect in rects if rect[2] >= min_size[0] and rect[3] > min_size[1])

    return rects


# def visualize_rects(img, rects): # 이미지에 bounding box 를 그려주는 함수
#     img_copy = img.copy()
#     for rect in rects:
#         (x, y, w, h) = rect
#         cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 3)
#     return img_copy
def visualize_rects(img, rect):
    x, y, w, h = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    return img

