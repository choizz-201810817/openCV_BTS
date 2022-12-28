#%%
import cv2
import numpy as np

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

        add_gray = cv2.cvtColor(add_img, cv2.COLOR_BGR2GRAY)
        roi = test_img[x:x+add_rows,y:y+add_cols] #로고파일 필셀값을 관심영역(ROI)으로 저장함

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
        test_img[x:x+add_rows,y:y+add_cols] = result_img

        result = test_img
        cv2.imshow(win_name, result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    

# %%
test_array=np.fromfile("./imgtrain/jk/0.jpg",np.uint8)
test_img=cv2.imdecode(test_array,-1)

add_array=np.fromfile("./imgtrain/jk/sign2.jpg",np.uint8)
add_img=cv2.imdecode(add_array,-1)


trackbar('kby',test_img, add_img)

# %%
