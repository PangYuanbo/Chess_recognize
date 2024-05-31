from pyk4a import PyK4A
import time
import cv2
from datetime import datetime

# 初始化 Kinect 摄像机
k4a = PyK4A()
k4a.start()
import os
import time
from datetime import datetime
import cv2
from pyk4a import PyK4A

k4a = PyK4A()
k4a.open()
k4a.start()

try:
    while True:
        # 获取捕捉
        capture = k4a.get_capture()
        if capture.color is not None:
            img_color = capture.color
            # 将 BGRA 转换为 RGB
            img_color = img_color[:, :, 2::-1]

            # 创建文件名和路径
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            # 使用当前目录下的一个文件夹来保存图片
            filename = f"{os.getcwd()}/{timestamp}.png"

            # 保存图片
            cv2.imwrite(filename, img_color)
            print(f"Saved {filename}")

        # 等待2秒
        time.sleep(2)

except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    # 停止摄像机
    k4a.stop()
    k4a.close()
