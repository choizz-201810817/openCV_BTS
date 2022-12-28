# %%
import os
import cv2
import pandas as pd

def Cutting_face_save(img, name, member=''):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cropped = img[y: y+h, x: x+w]
        resize = cv2.resize(cropped, (180,180))
        rotated90 = cv2.rotate(resize, cv2.ROTATE_90_CLOCKWISE)
        rotated270 = cv2.rotate(resize, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rotated180 = cv2.rotate(resize, cv2.ROTATE_180)
        # 이미지 저장하기
        cv2.imwrite(f"./cutting_faces/{member}/{member}{name}.png", resize)
        cv2.imwrite(f"./cutting_faces/{member}/{member}{name}_90.png", rotated90)
        cv2.imwrite(f"./cutting_faces/{member}/{member}{name}_180.png", rotated180)
        cv2.imwrite(f"./cutting_faces/{member}/{member}{name}_270.png", rotated270)

# %%
members = ['jhope', 'jimin', 'jin', 'rm', 'suga', 'v']

for member in members:
    files = os.listdir(f'./img_ori/{member}')

    for i, file in enumerate(files):
        img = cv2.imread(f"./img_ori/{member}/"+file)
        Cutting_face_save(img, i, member)

# %%
