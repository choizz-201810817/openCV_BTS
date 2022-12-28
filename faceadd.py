#%%
# 이미지 단순합성 
import cv2 
import numpy as np 
import matplotlib.pylab as plt 
# import sys 
#%%
# window생성 
win_name = 'add img' # 창이름 
trackbar_name = 'fade_change' # 트랙바 이름 

# 오류표시
# img = cv2.imread('lenna.bmp')
# if img is None:
#     print('Image load failed')
#     sys.exit()
# cv2.imshow('lenna', img)
# cv2.waitKey()

# cv2.destroyAllWindows()

# 트랙바 이벤트 핸들러 함수 
def onChange(x):
    alpha = x/100
    dst = cv2.addWeighted(img1, 1-alpha, img2, alpha, 0)
    cv2.imshow(win_name, dst)
    

# 이미지 불러오기 
img1 = cv2.imread('./img/6.png')

img2 = cv2.imread('./img/5_1.jpg')

cv2.imshow(win_name, img1)
cv2.createTrackbar(trackbar_name, win_name, 0 , 100, onChange)

cv2.waitKey()
cv2.destroyAllWindows()


# # 이미지 덧셈 
# # img3 = img1 + img2 # 더하기 연산 
# img4 = cv2.add(img1, img2) # opencv 함수
# # 'img1':img1, 'img2':img2, 'img1+img2': img3, 

# imgs = {'add sign': img4}

# # 이미지 출력 
# for i, (k, v) in enumerate(imgs.items()):
#     plt.subplot(2,2, i + 1)
#     plt.imshow(v[:, :, ::-1])
#     plt.title(k)
#     plt.xticks([]); plt.yticks([])
# plt.show()
# %%
