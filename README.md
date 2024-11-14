## 文件划分说明

```
|--pic (测试图片)
|--utils
  |--data.py (自定义数据类型函数库)
  |--detect_v5lite.py (yolov5_lite推理代码封装)
  |--detect_v8.py (yolov8推理代码封装)
|--video (测试视频)
|--weights (训练结果)
  |--competition.onnx (工创赛数据集，四大类生活垃圾，v5lite)
  |--gear.onnx (齿轮缺陷，四类，yolov8)
  |--insulator.onnx (insulator缺陷，一类， yolov8)
|--Main.py (yolov8+齿轮缺陷+QT综合使用案例)
|--v8.py (yolov8示例)
|--v5lite.py (yolov5-lite示例)
```

可以去我的文档网站学习模型训练：[YOLO模型训练](https://tonmoon.top/study/yolov5/1/)

## 克隆源码

打开Linux终端，克隆源码

```shell
git clone https://github.com/wxnlP/optimization.git
```

将<kbd>utils</kbd>放到自己的工程文件夹根目录（自备训练结果权重），安装依赖

```shell
#更新系统包列表
sudo apt update
#安装opencv-python
pip install opencv-python
#安装numpy
pip install numpy
#安装onnxruntime
pip install onnxruntime
#安装pyserial（如果需要通信可以安装）
pip install pyserial
#yolov8需要库
pip install ultralytics
```

## v5lite

### 初始化

根据自己的文件目录修改必要参数

```python
from utils.detect_v5lite import Detect

""" 对象初始化，对应传入参数如下
	label_value: "训练是列的标签，注意自己的顺序"
	label_key: "标签对应序列，数据类型为列表"
	model_path: "ONNX文件的目录，注意：Linux报错要换成绝对路径"
"""
# 示例如下，为垃圾分类标签
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
model_path = "weights/competition.onnx"
label_key = list(range(22))
APP = Detect(model_path=model_path, label_key=label_key, label_value=labels)
```

> 根据自己系统和目录修改路径
>

### yolov5_lite.py使用方法

将<kbd>utils</kbd>文件夹放到工程根目录

```python
# 续 “初始化” 后
# 照片识别，ch参数--1:返回DetectData数据类型--0:返回img数据（默认ch=0）
APP.detect_pic(img, ch=1)
# 视频处理，传入视频地址--默认输出视频存放地址--'/home/sunrise/DefectDetect/videos/output_video.mp4'
# 根据设备自行修改
APP.detect_video(path)
```

### 实时检测

```python
# 续 “初始化” 后
"""摄像头处理示例"""
video = 0
cap = cv2.VideoCapture(video)
while True:
    success, img = cap.read()
    if success:
        data = APP.detect_pic(img, ch=1)
        print("*"*50)
        data.show()
```

### 单个照片处理

```python
"""照片处理示例"""
img = cv2.imread(pic_path)
APP.detect_pic(img)
```

### 视频处理

```python
"""视频处理示例"""
APP.detect_video(video_path)
```

## v8

### 初始化

```python
from utils.detect_v8 import Detect

"""检测对象初始化"""
model_path = "/home/sunrise/v8/weights/gear.onnx"
label_dic = {
    0: "MissingTeeth",
    1: "Potholes",
    2: "Scratches",
    3: "Gear"
    }
# parts参数是单独为此工程场景定制【1】
part_label = [3]
APP = Detect(model_path=model_path, label_dic=label_dic, parts=part_label)
```

【1】: 因为数据集中`0-2`为缺陷，为`3`为产品名称，故为了排除`3`显示为缺陷，特定了`parts`参数，不适应这个功能可以忽略传入空列表即可。

### detect_v8使用

将<kbd>utils</kbd>文件夹放到工程根目录

```python
# pic_path: 图片路径，kd: 返回参数选择--1：自定义DetectData数据--0: 照片数据（cv2可读取）
APP.get_pic(pic_path, kd=1)
# 获取当前帧，照片数据（cv2可读取），便于同时获取 “自定义DetectData数据” “照片数据”
APP.get_capFrame()
# 
```

### 多照片推理

```python
"""多照片推理"""
pic_path = [
    "/home/lzh/optimization/pic/gear (1).jpg",
    "/home/lzh/optimization/pic/gear (2).jpg",
    "/home/lzh/optimization/pic/gear (3).jpg"
]
for i in pic_path:
    APP.get_pic(i)
    img = APP.get_capFrame()
```

### 视频推理

```python
"""视频推理"""
# 传入视频路径就可以，输出视频默认保存在"/home/sunrise/v8/video/output.mp4"，定位到get_video函数去修改
APP.get_video(video_path)
```

