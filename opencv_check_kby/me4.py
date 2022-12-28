#%%
#pip install opencv-contrib-python
import cv2
import numpy as np
from os import makedirs, listdir
from os.path import isdir, isfile, join

#%%
## 1. 얼굴을 검출해서 train 샘플 이미지 저장하는 함수 만들기


# 1-1. 얼굴 검출 함수
def get_face(img):

    # 얼굴인식 객체 생성
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # BGR 형식을 GRAY 형식으로 변환
    face_rect = face_classifier.detectMultiScale(gray, 1.3,5) # 사진에서 얼굴 검출
    
    # 얼굴이 인식 안되면 패스!
    if face_rect==():
        pass

    # 얼굴 인식 되면 얼굴 부위 크롭해서 리턴
    for(x,y,w,h) in face_rect:
        crop_face = img[y:y+h, x:x+w]
        return crop_face


# 1-2. 검출한 얼굴 파일로 저장
def train_save(name):
    
    # 샘플이미지 저장할 폴더 :: 해당 이름의 폴더가 없다면 생성
    if not isdir("imgtrain/"+name+"/train"):
        makedirs("imgtrain/"+name+"/train")

    # 처음 이미지가 저장되어 있는 곳에서 모든 이미지 파일 경로들을 리스트로 가져옴
    folder_path = "imgtrain/"+name
    path = [join(folder_path,f) for f in listdir(folder_path) if isfile(join(folder_path,f))]

    count=0
    for img_path in path:
        img = cv2.imread(img_path) # 이미지 불러오기

        # 얼굴 인식이 되면
        if get_face(img) is not None:
            count+=1    
            crop_face_resize = cv2.resize(get_face(img), (100,100)) # 이미지 크기 조절
            gray_face = cv2.cvtColor(crop_face_resize, cv2.COLOR_BGR2GRAY) # 이미지 흑백 전환

            # 학습할 이미지 저장
            file_name = "imgtrain/"+name+"/train/"+str(count)+".jpg"
            cv2.imwrite(file_name,gray_face)
        
        # 얼굴 인식이 안되면 패스
        else :
            pass

#%%
## 2. 이미지 샘플 저장하기(함수실행)
# folder_path = "imgtrain/"
# users = [f for f in listdir(folder_path) if isdir(join(folder_path,f))]
# users

train_save('jk')


# %%
## 3. 얼굴 학습 함수 생성

# 한명 얼굴 학습
def train(name):
    # name의 이미지 샘플(훈련할 이미지) 저장 경로
    folder_path = "imgtrain/"+name+"/train"

    # 샘플 이미지들 경로 리스트로 저장
    path = [join(folder_path,f) for f in listdir(folder_path) if isfile(join(folder_path,f))]

    Training_Data, Labels = [], [] # 샘플이미지와 인덱스 리스트 만들기
    for i, img_path in enumerate(path):
        images = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # 이미지가 아니면 패스
        if images==():
            continue

        Training_Data.append(np.asarray(images, dtype=np.uint8)) # 학습을 위해 배열화
        Labels.append(i)

    if len(Labels) == 0: # 학습할 샘플이미지가 아예 없으면 그냥 끝남
        print("There is no data to train.")
        return None

    Labels = np.asarray(Labels, dtype=np.int32)

    # 모델 생성 
    model = cv2.face.LBPHFaceRecognizer_create()
    # 학습
    model.train(np.asarray(Training_Data), np.asarray(Labels))
    print(name + " : Model Training Complete!!!!!")

    #학습 모델 리턴
    return model


# 여러명 얼굴 학습
def trains():
    # train 폴더내의 모든 폴더 가져옴(각 user 폴더), 리스트화
    folder_path = "imgtrain/"
    users = [f for f in listdir(folder_path) if isdir(join(folder_path,f))]
    
    #학습 모델 저장할 딕셔너리(유저명 : 유저확인하는 함수)
    models = {}

    # 각 폴더에 있는 얼굴들 학습
    for user in users:
        print('model :' + user)
        
        result = train(user) #학습
        if result ==(): # 학습이 안됐으면 패스
            continue

        print('model2 :' + user)
        models[user] = result # 학습되었으면 저장

    # 학습된 모델 딕셔너리 리턴
    return models    




# %%
## 4. 학습 시킨 것을 바탕으로 현재 얼굴이 누구인지 알아내기

# 4-1. 얼굴 인식 함수
def face_detector(img, size = 0.5):
    # 객체 생성
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    img_size = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 흑백전환
    faces = face_classifier.detectMultiScale(gray,1.3,5) # 얼굴검출

    # 이미지 검출안되면 사진만 리턴
    if faces ==():
        return img,img_size,[]

    # 이미지 검출되면 사각형 그리고, 이미지와 얼굴좌표 리턴
    rect = []
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (100,100))
        rect = [x,y,w,h]

    return img,img_size,rect   #검출된 좌표에 사각 박스 그리고(img), 검출된 부위를 잘라(roi) 전달


# 4-2. 트랙바로 로고 위치 변경 & 합성 & s키 두번 누르면 저장
def onChange(pos):
    pass
def trackbar(win_name,test_img,add_img):
    add_img = cv2.resize(add_img,(0,0),fx=0.4,fy=0.4)
    test_rows, test_cols, test_channels = test_img.shape
    add_rows, add_cols, add_channels = add_img.shape #로고파일 픽셀값 저장
   
    # 윈도우 창 생성
    # cv2.imshow(win_name, img)
    cv2.namedWindow(win_name)

    # 트랙바 생성(포스에서 소수점자리 안되므로 일단은 정수로 표현후, 추후에 sigma_r 100으로 나눔)
    cv2.createTrackbar('x', win_name, 0, test_rows-add_rows, onChange)
    cv2.createTrackbar('y', win_name, 0, test_cols-add_cols, onChange)

    # 트랙바 디폴트값 설정
    cv2.setTrackbarPos('x', win_name, 0)
    cv2.setTrackbarPos('y', win_name, 0)

    while True:
        # q누르면 종료
        if cv2.waitKey(1) == ord('q'):
            break

        # s누르면 사진 저장
        if cv2.waitKey(1) == ord('s'):
                file_name = "imgtrain/jk/add_sign.jpg"
                cv2.imwrite(file_name,result)
        
        try:

            x = cv2.getTrackbarPos('x', win_name)
            y = cv2.getTrackbarPos('y', win_name)
            
            try:
                test_copy_img = np.copy(test_img)
                add_gray = cv2.cvtColor(add_img, cv2.COLOR_BGR2GRAY)
                roi = test_copy_img[x:x+add_rows,y:y+add_cols] #로고파일 필셀값을 관심영역(ROI)으로 저장함

                # 계산된 문턱값과, 그것을 적용해서 나타낸 결과이미지(배경은 흰색으로, 그림을 검정색으로 변경)
                ret, mask = cv2.threshold(add_gray, 160, 255, cv2.THRESH_BINARY) 
                mask_inv = cv2.bitwise_not(mask) # mask영역에서의 보색 출력
                # mask 배경 흰색, 로고 검정 / mask_inv 배경 검정, 로고 흰색

                #배경에서만 연산 :: test_img 배경 복사
                test_bg = cv2.bitwise_and(roi,roi,mask=mask) 
                # 로고에서만 연산 :: add_img
                add_bg = cv2.bitwise_and(add_img,add_img,mask=mask_inv) 

                # test_bg, add_bg 합성
                result_img = cv2.bitwise_or(test_bg, add_bg)
                # test_img에 result_img 넣기
                test_copy_img[x:x+add_rows,y:y+add_cols] = result_img

                result = test_copy_img
                cv2.imshow(win_name, result)
            except:
                break
        except:
            break
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 4-3. 예측함수
def run(models, test_img, add_img):
    # test img에서 얼굴 인식하기
    test_copy_img = np.copy(test_img)

    img,img_size,rect = face_detector(test_img)
    face = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 얼굴 인식이 되면
    try:
        # 위에서 학습한 모델로 예측시도
        for user_id, model in models.items(): #confidence(신뢰도, 0에 가까울수록 학습한 해당 유저와 일치한다는 뜻)
            id, confidence = model.predict(face)
            acc_score =  int(100*(1-(confidence)/300)) # 0~100%로 정확도 나타내려고

        # 50%이상으로 해당 인물 확신하면, ()%로 해당인물인지 이미지에 나타내기
        if acc_score > 50:
            id = user_id
            cv2.putText(img,str(acc_score)+'% '+id , (rect[0]+5, rect[1]-5), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 0), 2) # 1: 글자 크기 확대 비율, 2: 글자두께
            cv2.imshow('face find_kby', img)

            # 사진에 add_img 합성
            trackbar('add_img',test_copy_img, add_img)

        # 아니면 모르는 사람이라고 이미지에 나타내기
        else :
            id = "unknown"
            cv2.putText(img,id, (rect[0]+5, rect[1]-5), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow('face find_kby', img)
            

    #얼굴 인식자체가 안되면, 얼굴인식 안된다고 이미지에 나타내기
    except:
        cv2.putText(img, "Face Not Found", (250, 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 50, 255), 2)
        cv2.imshow('face find_kby', img)
        pass
    
    cv2.waitKey(0)
    cv2.destroyAllWindows() 


# %%
test_img = cv2.imread("./imgtrain/jk/120.jpg")
add_img = cv2.imread("./imgtrain/jk/sign2.jpg")

if __name__ == "__main__":
    # 학습 시작
    models = trains()
    # 고!
    run(models, test_img, add_img)





#%%
import cv2
import numpy as np

# 트랙바로 로고 위치 변경 & 합성 & s키 두번 누르면 저장
def onChange(pos):
    pass
def trackbar(win_name,test_img,add_img):
    add_img = cv2.resize(add_img,(0,0),fx=0.4,fy=0.4)
    test_rows, test_cols, test_channels = test_img.shape
    add_rows, add_cols, add_channels = add_img.shape #로고파일 픽셀값 저장
   
    # 윈도우 창 생성
    # cv2.imshow(win_name, img)
    cv2.namedWindow(win_name)

    # 트랙바 생성(포스에서 소수점자리 안되므로 일단은 정수로 표현후, 추후에 sigma_r 100으로 나눔)
    cv2.createTrackbar('x', win_name, 0, test_rows-add_rows, onChange)
    cv2.createTrackbar('y', win_name, 0, test_cols-add_cols, onChange)

    # 트랙바 디폴트값 설정
    cv2.setTrackbarPos('x', win_name, 0)
    cv2.setTrackbarPos('y', win_name, 0)

    while True:
        # q누르면 종료
        if cv2.waitKey(1) == ord('q'):
            break

        # s누르면 사진 저장
        if cv2.waitKey(1) == ord('s'):
                file_name = "imgtrain/jk/add_sign.jpg"
                cv2.imwrite(file_name,result)

        x = cv2.getTrackbarPos('x', win_name)
        y = cv2.getTrackbarPos('y', win_name)
        
        try:
            test_copy_img = np.copy(test_img)
            add_gray = cv2.cvtColor(add_img, cv2.COLOR_BGR2GRAY)
            roi = test_copy_img[x:x+add_rows,y:y+add_cols] #로고파일 필셀값을 관심영역(ROI)으로 저장함

            # 계산된 문턱값과, 그것을 적용해서 나타낸 결과이미지(배경은 흰색으로, 그림을 검정색으로 변경)
            ret, mask = cv2.threshold(add_gray, 160, 255, cv2.THRESH_BINARY) 
            mask_inv = cv2.bitwise_not(mask) # mask영역에서의 보색 출력
            # mask 배경 흰색, 로고 검정 / mask_inv 배경 검정, 로고 흰색

            #배경에서만 연산 :: test_img 배경 복사
            test_bg = cv2.bitwise_and(roi,roi,mask=mask) 
            # 로고에서만 연산 :: add_img
            add_bg = cv2.bitwise_and(add_img,add_img,mask=mask_inv) 

            # test_bg, add_bg 합성
            result_img = cv2.bitwise_or(test_bg, add_bg)
            # test_img에 result_img 넣기
            test_copy_img[x:x+add_rows,y:y+add_cols] = result_img

            result = test_copy_img
            cv2.imshow(win_name, result)
        except:
            break
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    

# %%
test_array=np.fromfile("./imgtrain/jk/0.jpg",np.uint8)
test_img=cv2.imdecode(test_array,-1)

add_array=np.fromfile("./imgtrain/jk/sign2.jpg",np.uint8)
add_img=cv2.imdecode(add_array,-1)


trackbar('kby',test_img, add_img)

# %%
