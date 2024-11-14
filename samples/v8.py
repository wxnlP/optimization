from utils.detect_v8 import Detect

model_path = "/home/sunrise/v8/weights/insulator.onnx"
pic_path = [
    "/home/sunrise/v8/pic/0049.jpg",
    "/home/sunrise/v8/pic/0082.jpg",
    "/home/sunrise/v8/pic/0120.jpg",
    "/home/sunrise/v8/pic/0122.jpg",
    "/home/sunrise/v8/pic/0164.jpg",
    "/home/sunrise/v8/pic/0183.jpg"
    ]
label_dic = {0: "instulator"}
app = Detect(model_path=model_path, label_dic=label_dic)
"""多照片推理"""
# for i in pic_path:
#     app.get_pic(i)
"""视频推理"""
app.get_cap()