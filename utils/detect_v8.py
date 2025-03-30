from utils.data import DetectData
from ultralytics import YOLO
import cv2

class DetectData:
    def __init__(self, parts,labels):
        """ 自定义检测数据类型:
        name = 零件名称
        kind = 缺陷类别
        coordinate_x = x坐标
        coordinate_y = y坐标
        confidence = 置信度
        """
        self.label_dic = labels
        self.parts = parts
        self.name = None
        self.num = None
        self.kind = []
        self.coordinate_x = []
        self.coordinate_y = []
        self.confidence = []
    
    def add_properties(self, number, kind, confidence, coordinate_x, coordinate_y):
        self.clear_properties()
        self.num = number
        for i in range(number):
            if int(kind[i]) not in self.parts:
                self.kind.append(self.label_dic[int(kind[i])])
                self.coordinate_x.append(coordinate_x[i]) 
                self.coordinate_y.append(coordinate_y[i])
                self.confidence.append(confidence[i])
            else:
                self.name = self.label_dic[int(kind[i])]
                self.num = self.num-1
    
    def show_properties(self):
        print(f"物件名称--{self.name}")
        for i in range(self.num):
            print(f"{'##'}缺陷类型--{self.kind[i]}")
            print(f"{'##'*2}中心坐标X--{self.coordinate_x[i]}")
            print(f"{'##'*2}中心坐标Y--{self.coordinate_y[i]}")
            print(f"{'##'*2}置信度--{self.confidence[i]}")


class Detect():
    def __init__(self, model_path, label_dic, parts, img_size = 320):
        self.img_size = img_size
        self.model = YOLO(model_path)
        self.label = label_dic
        self.parts = parts
        self.data = DetectData(self.parts, self.label)
        self.frame=None
       
    def get_pic(self, pic, kd=0):
        """获取照片推理结果"""
        results = self.model(pic, stream=True, imgsz=self.img_size)
        for result in results:
            boxes = result.boxes
            print("数量", boxes.cls.numel())
            if boxes.cls.numel():
                # xyxy坐标处理
                xy = boxes.xyxy
                coordinate_x = []
                coordinate_y = []
                for i in range(boxes.cls.numel()):
                    xy = boxes.xyxy[i]
                    x1 = xy[0]
                    y1 = xy[1]
                    x2 = xy[2]
                    y2 = xy[3]
                    coordinate_x.append((x1 + x2)*0.5)
                    coordinate_y.append((y1 + y2)*0.5)
                self.data.add_properties(boxes.cls.numel(), boxes.cls, confidence=boxes.conf, coordinate_x=coordinate_x, coordinate_y=coordinate_y)

            else:
                print("未检测到")
            self.frame = result.plot()
        self.data.show_properties()

        if kd:
            return self.data
        else:
            return self.frame
            
    def get_capFrame(self):
        return self.frame

    def get_video(self, video_path):
        """推理视频"""
        # 读取视频
        cap = cv2.VideoCapture(video_path)  
        # 获取视频的宽度和高度
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 设置视频写入器，设置编码器为XVID，帧率为30fps
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 也可以使用'X264'等其他编码
        out = cv2.VideoWriter("/home/sunrise/v8/video/output.mp4", fourcc, 30.0, (width, height))
        while cap.isOpened():
            # 读取帧
            success, frame = cap.read()
            if success:
                # 推理帧
                results = self.model(frame, stream=True, imgsz=self.img_size)
                for result in results:
                    self.frame = result.plot()
                # 将标注后的帧写入视频文件
                out.write(self.frame)
            else:
                break
                
            
