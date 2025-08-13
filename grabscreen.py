import cv2
import numpy as np
from mss import mss

def grab_screen(region=None):
    with mss() as sct:
        if region:
            monitor = {"top": region[1], "left": region[0], "width": region[2]-region[0] + 1, "height": region[3]-region[1] + 1}
        else:
            monitor = sct.monitors[0]  # 默认主屏幕
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # 转为 BGR（去掉 Alpha 通道）
        return img

if __name__ == '__main__':
    # 示例：截取屏幕 (100, 100) 到 (500, 500) 的区域
    screen = grab_screen(region=(100, 100, 500, 500))
    print(screen.shape)
    cv2.imshow("Screen Capture", screen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()