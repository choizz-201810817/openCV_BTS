#%% 
#python -m pip install opencv-python
## 1. 얼굴과 눈 부분 인식하기
import cv2

# 얼굴인식, 눈인식 xml 객체 생성 & 잘 생성됐는지 확인
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
if face_cascade.empty() | eye_cascade.empty() :
    print('XML load failed!')
else :
    print('XML load success!')
    
# 사진 가져옴 & 잘 가져왔는지 확인
img = cv2.imread('./img/0.jpg')
if img is None:
    print('Image load failed!')
else :
    print('Image load success!')
    
# (참고) img 있으면 => len(img)>=로 표현가능

# BGR 형식을 GRAY 형식으로 변환
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 입력영상에서 얼굴 검출
faces = face_cascade.detectMultiScale(gray, 1.3,5)

#위의 함수 설명
#cv2.CascadeClassifier.detectMultiScale(image, scaleFactor=None, minNeighbors=None, flags=None, minSize=None, maxSize=None)
#image: 입력 영상 (cv2.CV_8U)
#scaleFactor: 영상 축소 비율. 기본값은 1.1.
#minNeighbors: 얼마나 많은 이웃 사각형이 검출되어야 최종 검출 영역으로 설정할지를 지정. 기본값은 3.
#flags: (현재) 사용되지 않음
#minSize: 최소 객체 크기. (w, h) 튜플.
#maxSize: 최대 객체 크기. (w, h) 튜플.
#result: 검출된 객체의 사각형 정보(x, y, w, h)를 담은 numpy.ndarray. shape=(N, 4). dtype=numpy.int32.


# 얼굴 좌표 리턴
for (x, y, w, h) in faces:
    
    # 얼굴 좌표로 사격형 그림
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # 눈 검출은 얼굴이 검출된 영역 내부에서만 진행하기 위해 얼굴 좌표 가져옴
    roi_gray = gray[y : y + h, x : x + w]
    roi_color = img[y : y + h, x : x + w]
    
    # 눈 검출 좌표 리턴
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
    
    # roi_color에 표시 눈 좌표 사각형 그려서 원본에도 표시
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

# 결과 보여주기
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()





#%%
## 2. 얼굴 찾아서 모자이크 처리
import cv2

# 얼굴인식, 눈인식 xml 객체 생성 & 잘 생성됐는지 확인
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
if face_cascade.empty() | eye_cascade.empty() :
    print('XML load failed!')
else :
    print('XML load success!')
    
# 사진 가져옴 & 잘 가져왔는지 확인
img = cv2.imread('./img/0.jpg')
if img is None:
    print('Image load failed!')
else :
    print('Image load success!')

# BGR 형식을 GRAY 형식으로 변환
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 입력영상에서 얼굴을 검출
faces = face_cascade.detectMultiScale(gray, 1.3,5)

 # 얼굴 좌표 리턴
for (x, y, w, h) in faces:
    
    # 얼굴 좌표로 사격형 그림
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # 얼굴을 모자이크 처리할 것이므로, 얼굴 좌표 가져옴
    roi_color = img[y : y + h, x : x + w]
    
    # resize로 이미지 크기 및 모자이크 처리 (dsize로 크기 조정, interpolation으로 보간법 지정)
    roi = cv2.resize(roi_color, dsize=(0,0), fx=0.05, fy=0.05) 
    roi = cv2.resize(roi, (w, h), interpolation=cv2.INTER_AREA)
    img[y:y+h, x:x+w] = roi #적용

# 결과 보여주기
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()





# %%
## 3. 동영상 파일에서 얼굴과 눈 검출하기
import cv2

# 얼굴인식, 눈인식 xml 객체 생성 & 잘 생성됐는지 확인
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye.xml')
if face_cascade.empty() | eye_cascade.empty() :
    print('XML load failed!')
else :
    print('XML load success!')
    
# 영상 가져옴 & 잘 가져왔는지 확인
cap = cv2.VideoCapture('./img/bts_sample.avi')
if cap is None:
    print('video load failed!')
else :
    print('video load success!')

while(cap.isOpened()):
    # 동영상을 가져오고 & 잘 가져오고 있는지 확인
    ret, frame = cap.read()
    if not ret:
        print('video read failed!')
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 동영상을 회색으로 변환
    faces = face_cascade.detectMultiScale(gray, 1.1, 10) #동영상에서 얼굴 추출
    
    for (x, y, w, h) in faces:
        # 얼굴 좌표로 사격형 그림
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # 얼굴을 모자이크 처리할 것이므로, 얼굴 좌표 가져옴
        roi_color = frame[y : y + h, x : x + w]
        
        # resize로 이미지 크기 및 모자이크 처리 (dsize로 크기 조정, interpolation으로 보간법 지정)
        roi = cv2.resize(roi_color, dsize=(0,0), fx=0.05, fy=0.05) 
        roi = cv2.resize(roi, (w, h), interpolation=cv2.INTER_AREA)
        frame[y:y+h, x:x+w] = roi #적용
    
    cv2.imshow('frame', frame)
    
    # q누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()









# %%
## 4. 동영상 파일에서 얼굴 모자이크처리하기
import cv2

# 얼굴인식, 눈인식 xml 객체 생성 & 잘 생성됐는지 확인
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye.xml')
if face_cascade.empty() | eye_cascade.empty() :
    print('XML load failed!')
else :
    print('XML load success!')
    
# 영상 가져옴 & 잘 가져왔는지 확인
cap = cv2.VideoCapture('./img/bts_sample.avi')
if cap is None:
    print('video load failed!')
else :
    print('video load success!')

while(cap.isOpened()):
    # 동영상을 가져오고 & 잘 가져오고 있는지 확인
    ret, frame = cap.read()
    if not ret:
        print('video read failed!')
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 동영상을 회색으로 변환
    faces = face_cascade.detectMultiScale(gray, 1.1, 10) #동영상에서 얼굴 추출
    
    for (x, y, w, h) in faces:
        # 얼굴 좌표로 사격형 그림
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # 눈 검출은 얼굴이 검출된 영역 내부에서만 진행하기 위해 얼굴 좌표 가져옴
        roi_gray = gray[y : y + h, x : x + w]
        roi_color = frame[y : y + h, x : x + w]

        # 눈 검출 좌표 리턴
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)

        # roi_color에 표시 눈 좌표 사각형 그려서 원본에도 표시
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    
    cv2.imshow('frame', frame)
    
    # q누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()





#%%
## 5. 사진을 그림으로 변경
import numpy as np
import cv2

# 사진 가져옴 & 잘 가져왔는지 확인
img = cv2.imread('./img/0.jpg')
if img is None:
    print('Image load failed!')
else :
    print('Image load success!')

# sigma_s는 이미지가 얼마나 스무스해질지, sigma_r은 이미지가 스무스해지는 동안 뷰를 일정하게 유지하고 목적지를 지정
cartoon_img = cv2.stylization(img, sigma_s=100, sigma_r=0.05)

cv2.imshow('cartton view', cartoon_img)
cv2.waitKey(0)
cv2.destroyAllWindows()





# %%
## 6. sigma_s, sigma_r에 따라서 그림화되는 정도가 변화하는 것 만들어보기
import numpy as np
import cv2

# 사진 가져옴 & 잘 가져왔는지 확인
img = cv2.imread('./img/0.jpg')
if img is None:
    print('Image load failed!')
else :
    print('Image load success!')

def onChange(pos):
    pass

# 윈도우 창 생성
cv2.namedWindow('Trackbar Windows')

# 트랙바 생성(포스에서 소수점자리 안되므로 일단은 정수로 표현후, 추후에 sigma_r 100으로 나눔)
# cv2.createTrackbar("트랙 바 이름", "윈도우 창 제목", 최솟값, 최댓값, 콜백 함수)
cv2.createTrackbar('sigma_s', 'Trackbar Windows', 0, 200, onChange)
cv2.createTrackbar('sigma_r', 'Trackbar Windows', 0, 100, onChange)

# 트랙바 디폴트값 설정
#cv2.setTrackbarPos("트랙 바 이름", "윈도우 창 제목", 설정값)
cv2.setTrackbarPos('sigma_s', 'Trackbar Windows', 100)
cv2.setTrackbarPos('sigma_r', 'Trackbar Windows', 10)

while True:

    # q누르면 종료
    if cv2.waitKey(1) == ord('q'):
        break

    # 트랙바 값 받아서 지정(getTrackbarPos(트랙바 이름, 윈도우 창 제목))
    sigma_s_value = cv2.getTrackbarPos('sigma_s', 'Trackbar Windows')
    sigma_r_value = cv2.getTrackbarPos('sigma_r', 'Trackbar Windows')/100

    # img를 그림으로 변화
    cartoon_img = cv2.stylization(img, sigma_r=sigma_r_value, sigma_s=sigma_s_value)

    # 그림 보이기
    cv2.imshow('Trackbar Windows', cartoon_img)

cv2.destroyAllWindows()





# %%
