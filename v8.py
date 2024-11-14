from utils.detect_v8 import Detect
import cv2

"""检测对象初始化"""
model_path = "/home/lzh/optimization/weights/gear.onnx"
label_dic = {
    0: "MissingTeeth",
    1: "Potholes",
    2: "Scratches",
    3: "Gear"
    }
# parts参数是单独为此工程场景定制
part_label = [3]
APP = Detect(model_path=model_path, label_dic=label_dic, parts=part_label)
"""多照片推理"""
pic_path = [
    "/home/lzh/optimization/pic/gear (1).jpg",
    "/home/lzh/optimization/pic/gear (2).jpg",
    "/home/lzh/optimization/pic/gear (3).jpg"
]
for i in pic_path:
    APP.get_pic(i)
    img = APP.get_capFrame()
"""视频推理"""
#APP.get_video(video_path)