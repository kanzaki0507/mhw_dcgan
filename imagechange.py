import cv2
import os

if __name__ == '__main__':
    def imagechange():
        names = os.listdir('./img/')
        names.sort()
        save_path = "img32"
        os.makedirs(save_path,exist_ok=True)
        for name in names:
            os.makedirs(save_path,exist_ok=True)
            img = cv2.imread('./img/' + name)
            width, height = 32, 32
            img = cv2.resize(img, (width, height))
            cv2.imwrite('./img32/' + name, img)

    imagechange()