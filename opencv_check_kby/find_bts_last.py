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
    
    # 샘플사진 저장할 폴더 :: 해당 이름의 폴더가 없다면 생성
    if not isdir("imgtrain/"+name+"/train"):
        makedirs("imgtrain/"+name+"/train")

    # 처음 사진이 저장되어 있는 곳에서 모든 사진 파일 경로들을 리스트로 가져옴
    folder_path = "imgtrain/"+name
    path = [join(folder_path,f) for f in listdir(folder_path) if isfile(join(folder_path,f))]

    count=0
    for img_path in path:
        img = cv2.imread(img_path) # 사진 가져오기

        # 얼굴 인식이 되면
        if get_face(img) is not None:
            count+=1    
            crop_face_resize = cv2.resize(get_face(img), (100,100)) # 사진 크기 조절
            gray_face = cv2.cvtColor(crop_face_resize, cv2.COLOR_BGR2GRAY) # 사진 흑백 전환

            # 학습할 샘플 사진 저장
            file_name = "imgtrain/"+name+"/train/"+str(count)+".jpg"
            cv2.imwrite(file_name,gray_face)
        
        # 얼굴 인식이 안되면 패스
        else :
            pass

#%%
## 2. 이미지 샘플 저장하기(함수실행)
train_save('jk')


# %%
## 3. 얼굴 학습 함수 생성

# 한명 얼굴 학습
def train(name):
    # name의 샘플(훈련할 사진) 저장 경로
    folder_path = "imgtrain/"+name+"/train"

    # 샘플 사진들 경로 리스트로 저장
    path = [join(folder_path,f) for f in listdir(folder_path) if isfile(join(folder_path,f))]

    Training_Data, Labels = [], [] # 샘플사진과 레이블 리스트 만들기
    for i, img_path in enumerate(path): # 샘플 사진들 하나씩 가져와서
        images = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # 읽어오고
        
        if images==(): # 사진이 아니면
            continue # 패스

        Training_Data.append(np.asarray(images, dtype=np.uint8)) # 학습을 위해 이미지 배열화해서 리스트에 넣기
        Labels.append(i) # 사진 레이블 추가

    if len(Labels) == 0: # 학습할 샘플사진이 없으면
        print("There is no data to train.") # 학습할 사진 없다고 출력하고
        return None # 종료

    Labels = np.asarray(Labels, dtype=np.int32) # 학습을 위해 레이블도 배열화

    model = cv2.face.LBPHFaceRecognizer_create() # 모델 생성
    model.train(np.asarray(Training_Data), np.asarray(Labels)) # 모델 학습
    print(name + " : Model Training Complete!!!!!")

    return model  #학습 모델 리턴


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
        
        result = train(user) # 학습
        if result ==(): # 학습이 안됐으면 패스
            continue

        print('model2 :' + user)
        models[user] = result # 학습되었으면 저장

    return models # 학습된 모델 딕셔너리 리턴




# %%
## 4. 학습 시킨 것을 바탕으로 현재 얼굴이 누구인지 알아내기

# 4-1. 얼굴 인식 함수
def face_detector(img, size = 0.5):
    
    # 얼굴 인식 객체 생성
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    img_size = img.shape[:2] # 사진 크기 가져옴(가로, 세로)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 흑백전환
    faces = face_classifier.detectMultiScale(gray,1.3,5) # 얼굴 인식

    if faces ==():  # 얼굴 인식 안되면
        return img,img_size,[] # 사진과 사진크기만 리턴

    # 얼굴 인식 되면
    rect = [] #얼굴 좌표 저장할 리스트 생성
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2) # 얼굴부분에 사각형 그리고
        rect = [x,y,w,h] # 얼굴좌표 저장

    return img,img_size,rect # 사진과, 사진크기, 얼굴좌표 리턴


# 4-2. 트랙바로 로고 위치 변경 & 합성 & s키 누르면 저장
def onChange(pos): 
    pass
def trackbar(win_name,test_img,add_img):
    add_img = cv2.resize(add_img,(0,0),fx=0.4,fy=0.4) # 로고 사이즈 조절
    test_rows, test_cols, test_channels = test_img.shape # 사진 가로,세로 길이 가져옴
    add_rows, add_cols, add_channels = add_img.shape # 로고 가로, 세로 길이 가져옴

    cv2.imshow(win_name, test_img) # 윈도우 창 생성
    
    cv2.createTrackbar('x', win_name, 0, test_rows-add_rows, onChange) # 트랙바 생성
    cv2.createTrackbar('y', win_name, 0, test_cols-add_cols, onChange)
    cv2.setTrackbarPos('x', win_name, 0) # 트랙바 디폴트값 설정
    cv2.setTrackbarPos('y', win_name, 0)

    while True:
        try: # 변동되는 트랙바 값을 가져옴
            x = cv2.getTrackbarPos('x', win_name)
            y = cv2.getTrackbarPos('y', win_name)
            
            try: # 트랙바 값이 변동될 때마다 사진 합성 새롭게 함
                test_copy_img = np.copy(test_img) # 사진 복사(로고 위치 변동될때마다 깨끗한 사진 필요하므로)
                add_gray = cv2.cvtColor(add_img, cv2.COLOR_BGR2GRAY) # 로고이미지 흑백화
                roi = test_copy_img[x:x+add_rows,y:y+add_cols] # 현재 사진에서 로고 크기만큼 관심 부분 지정

                # 계산된 문턱값과, 그것을 적용해서 나타낸 결과이미지(배경은 흰색으로, 그림을 검정색으로 변경)
                ret, mask = cv2.threshold(add_gray, 160, 255, cv2.THRESH_BINARY) # 문턱 값 보다 크면 value 작으면 0
                mask_inv = cv2.bitwise_not(mask) # mask영역에서의 보색 출력
                # mask 배경 흰색, 로고 검정 / mask_inv 배경 검정, 로고 흰색
                
                test_bg = cv2.bitwise_and(roi,roi,mask=mask) # 배경에서만 연산 :: test_img 배경 복사
                add_bg = cv2.bitwise_and(add_img,add_img,mask=mask_inv) # 로고에서만 연산 :: add_img

                result_img = cv2.bitwise_or(test_bg, add_bg) # test_bg, add_bg 합성
                test_copy_img[x:x+add_rows,y:y+add_cols] = result_img # test_img에 result_img 넣기

                result = test_copy_img 
                cv2.imshow(win_name, result) # 로고 합성한 이미지 보여주기
            except:
                break
        except:
            break
        
        # q누르면 종료
        if cv2.waitKey(1) == ord('q'):
            break

        # s누르면 사진 저장
        if cv2.waitKey(1) == ord('s'):
            file_name = "imgtrain/jk/add_sign.jpg" # 파일경로 및 파일명 설정
            cv2.imwrite(file_name,result) # 저장
            
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 4-3. 예측함수
def run():
    if (e1.get()=='')|(e2.get()==''): # tkinter에서 테스트사진 파일명, 로고사진 파일명 안 적으면
        tkinter.messagebox.showerror('error','파일명을\n입력하세요') # 경고창 발생
    else:
        try: # 경로 잘 입력했으면 
            test_file_path = "imgtrain/jk/"+e1.get()+".jpg" # 테스트사진 파일명 가져옴
            add_file_path = "imgtrain/jk/"+e2.get()+".jpg" # 로고사진 파일명 가져옴

            test_array=np.fromfile(test_file_path,np.uint8) # 테스트사진 가져옴
            test_img=cv2.imdecode(test_array,-1)

            add_array=np.fromfile(add_file_path,np.uint8) # 로고 사진 가져옴
            add_img=cv2.imdecode(add_array,-1)

            models = trains() # 학습

            # test img에서 얼굴 인식하기
            test_copy_img = np.copy(test_img) # 사진 복사(50% 얼굴인식 사각형 제거한 사진 필요하므로)
            img,img_size,rect = face_detector(test_img) # 얼굴인식 함수 사용해서 사진과 얼굴좌표 가져옴
            face = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 사진 흑백화

            try: # 얼굴 인식이 되면, 위에서 학습한 모델로 예측시도
                for user_id, model in models.items(): # confidence(신뢰도, 0에 가까울수록 학습한 해당 유저와 일치한다는 뜻)
                    id, confidence = model.predict(face) # 예측했을 때, 이름과 confidence 가져옴
                    acc_score =  int(100*(1-(confidence)/300)) # 0~100%로 정확도 나타내려고

                if acc_score > 50:
                    id = user_id # id는 추정 인물명으로 정의
                    # "()% id"라고 얼굴인식 사각형 위에 글씨 표시
                    cv2.putText(img,str(acc_score)+'% '+id , (rect[0]+5, rect[1]-5), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 0), 2) 
                    cv2.imshow('FIND FACE (BY.TEAM Bang Sihyuk)', img) # 창 보여주기

                    # 사진에 로고 사진 합성(트랙바로 로고 사진 위치 조절하여 저장하는 창)
                    trackbar('FACE ADD LOGO (BY.TEAM Bang Sihyuk)',test_copy_img, add_img)

                else :  # 학습했던 인물일 확률이 50% 이하면
                    id = "unknown" # id는 모르는 사람
                    # "id=unknown"으로 얼굴인식 사각형 위에 글씨 표시
                    cv2.putText(img,id, (rect[0]+5, rect[1]-5), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.imshow('Find Face (BY.TEAM Bang Sihyuk)', img) # 창 보여주기

            except: #얼굴 인식 자체가 안되면
                # "Face Not Found"라고 특정 위치에 글씨 표시
                cv2.putText(img, "Face Not Found", (250, 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 50, 255), 2)
                cv2.imshow('FIND FACE (BY.TEAM Bang Sihyuk)', img) # 창 보여주기
                pass
            
            cv2.waitKey(0) # 무한대기
            cv2.destroyAllWindows() #창 종료

        except: # tkinter에서 해당 폴더에 테스트사진파일명, 로고사진파일명 없으면
            tkinter.messagebox.showerror('error','올바른 파일명을\n입력하세요') # 경고창 발생
    
def reset(): # tkinter에서 텍스트창들 리셋하는 함수
    e1.delete(0, 'end')
    e2.delete(0, 'end')


#%%
## 5. tkinter 실행
from tkinter import *
import tkinter.messagebox
window=Tk()
window.geometry('375x175+100+500') # tkinter창 크기 조절
window.title('FIND BTS (BY.TEAM 방시혁)') # tkinter창 이름 지정
# 레이블 배치
l3 = Label(window, text='', width=20)
l4 = Label(window, text='    실행 버튼을 누르고, 해당 이미지 저장을 원하는 경우\n    해당창을 클릭하고 \'Ctrl + S\'를 눌러주세요', width=50)
l5 = Label(window, text='', width=20)
l1 = Label(window, text='테스트 이미지 파일명', width=20)
l2 = Label(window, text='합성 이미지 파일명', width=20)
l3.grid(row=0, column=0)
l4.grid(row=1, column=0, columnspan=2)
l5.grid(row=2, column=0)
l1.grid(row=3, column=0)
l2.grid(row=4, column=0)

# 텍스트 입력 창
e1=Entry(window, width=30)
e2=Entry(window, width=30)
e1.grid(row=3, column=1)
e2.grid(row=4, column=1)

# 버튼 만들기
b1 = Button(window, text='실행', command=run,  width=13)
b2= Button(window, text='초기화', command=reset,  width=13)
b1.grid(row=5, column=1,  sticky='w')
b2.grid(row=5, column=1,  sticky='e')

window.mainloop()



# %%
