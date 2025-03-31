from utils.detect_v5lite import Detect
import cv2

"""对象初始化"""
labels = [
    "0-plastic_bottle",
    "0-drink_can",
    "0-paper",
    "0-carton",
    "0-milkCarton",
    "1-pericarp",
    "1-vegetable_leaf",
    "1-radish",
    "1-potato",
    "1-fruits",
    "2-battery",
    "2-Expired_drug",
    "2-button cell",
    "2-thermometer",
    "3-tile",
    "3-cobblestone",
    "3-brick",
    "3-paperCup",
    "3-tableware",
    "3-chopsticks",
    "3-butt",
    "3-mask"]
model_path = "/home/pi/optimization/weights/competition.onnx"
label_key = list(range(22))
APP = Detect(model_path=model_path, label_key=label_key, label_value=labels)

"""摄像头处理示例"""
video = 0
cap = cv2.VideoCapture(video)
while True:
    success, img = cap.read()
    if success:
        data = APP.detect_pic(img)
        print("*" * 50)
        # 返回值使用方法
        print(data[0].name)

"""照片处理示例"""
# img = cv2.imread("/home/pi/optimization/pic/image.png")
# data = APP.detect_pic(img)
# cv2.imwrite('/home/pi/optimization/results/result_v5lite.jpg', img)
# print(data)


"""视频处理示例"""
# cap = cv2.VideoCapture('/home/sunrise/DefectDetect/videos/2.mp4')
# APP.detect_video('/home/sunrise/DefectDetect/videos/2.mp4')
