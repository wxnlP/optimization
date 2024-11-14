from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from utils.data import DetectData
from utils.detect_v8 import Detect
import numpy as np
import time
import cv2
import sys
from PyQt5.QtCore import QThread, pyqtSignal

"""检测对象初始化"""
model_path = "/home/sunrise/v8/weights/gear.onnx"
label_dic = {
    0: "MissingTeeth",
    1: "Potholes",
    2: "Scratches",
    3: "Gear"
    }
part_label = [3]

APP = Detect(model_path=model_path, label_dic=label_dic, parts=part_label)


class VideoProcessingThread(QThread):
    update_signal = pyqtSignal(str)

    def __init__(self, file_name):
        super(VideoProcessingThread, self).__init__()
        self.file_name = file_name

    def run(self):
        self.update_signal.emit("视频处理中...")  # 更新 UI 显示正在处理
        APP.get_video(self.file_name)
        self.update_signal.emit("视频处理完成")  # 处理完成后更新显示



class UIWidgets(QMainWindow):
    def __init__(self):
        super(UIWidgets, self).__init__()
        self.button_begin = QPushButton("Detect-ing")
        self.button_stop = QPushButton("Stop-ed")
        self.button_sources = QPushButton("Sources...")
        self.button_clear = QPushButton("Clear")

        self.header_labels = ["零件名称", "缺陷类别", "中心坐标", "置信度"]
        self.table_data = QTableWidget(0, 4)
        self.table_data.setHorizontalHeaderLabels(self.header_labels)

        self.table_data.horizontalHeader().setStretchLastSection(True)
        for i in range(4):
            self.table_data.horizontalHeader().setSectionResizeMode(i, QHeaderView.Stretch)

        self.label_video = QLabel(self)
        self.capture = None
        self.timer = None  # 定时器实例
        self.label_video.setScaledContents(True)
        # 设置文本的位置
        self.label_video.setAlignment(Qt.AlignCenter)  # 文本居中显示
        # 也可以设置文字颜色，例如白色：
        self.label_video.setStyleSheet("color: red; font-size: 18px; font-weight: bold;")
        self.label_video.setText("等待操作...")

        self.layoutInit()
        self.signals()


    def layoutInit(self):
        layout_button1 = QHBoxLayout()
        layout_button2 = QHBoxLayout()
        layout1 = QVBoxLayout()
        layout2 = QHBoxLayout()

        layout_button1.addWidget(self.button_begin)
        layout_button1.addWidget(self.button_stop)
        layout_button2.addWidget(self.button_sources)
        layout_button2.addWidget(self.button_clear)

        layout1.addWidget(self.label_video)
        layout1.addLayout(layout_button1)
        layout1.addLayout(layout_button2)

        layout2.addWidget(self.table_data)
        layout2.addLayout(layout1)
        layout2.setStretch(0, 3)
        layout2.setStretch(1, 7)

        central_widget = QWidget()
        central_widget.setLayout(layout2)
        self.setCentralWidget(central_widget)

    def add_data(self, detect_data: DetectData):
        row_position = self.table_data.rowCount()
        for i in range(detect_data.num):
            self.table_data.insertRow(row_position)
            self.table_data.setItem(row_position, 0, QTableWidgetItem(detect_data.name))
            self.table_data.setItem(row_position, 1, QTableWidgetItem(detect_data.kind[i]))
            self.table_data.setItem(row_position, 2, QTableWidgetItem(f"X:{detect_data.coordinate_x[i]}/Y:{detect_data.coordinate_y[i]}"))
            self.table_data.setItem(row_position, 3, QTableWidgetItem(f"{detect_data.confidence[i]}"))
            self.table_data.scrollToBottom()

    def upload_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "选择文件", "", "Images (*.png *.jpg *.jpeg *.gif);;Videos (*.mp4 *.avi)", options=options)
        if file_name:
            file_extension = file_name.split('.')[-1].lower()
            if file_extension in ['png', 'jpg', 'jpeg', 'gif']:
                img = cv2.imread(file_name)
                if img is not None:
                    target_size = (640, 480)
                    img = cv2.resize(img, target_size)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    height, width, channel = img.shape
                    bytes_per_line = channel * width
                    data = APP.get_pic(img, kd=1)
                    self.add_data(data)
                    img = APP.get_capFrame()
                    q_image = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
                    self.label_video.setPixmap(QPixmap.fromImage(q_image))
                else:
                    QMessageBox.warning(self, "错误", "无法读取图像文件。")
            elif file_extension in ['mp4', 'avi']:
                self.label_video.setText("视频处理中...")
                # 创建并启动线程处理视频
                self.video_thread = VideoProcessingThread(file_name)
                # self.video_thread.update_signal.connect(self.update_label_video)
                self.video_thread.start()

    def signals(self):
        self.button_sources.clicked.connect(self.upload_file)
        self.button_begin.clicked.connect(self.start_camera)
        self.button_stop.clicked.connect(self.stop_camera)
        self.button_clear.clicked.connect(self.label_clear)

    def start_camera(self):
        self.start_timer()

    def stop_camera(self):
        if self.capture is not None:
            self.capture.release()
        self.stop_timer()
        self.label_clear()
                 
    def label_clear(self):
        self.label_video.clear()
        self.label_video.setText("等待操作...")

    def start_timer(self):
        # 创建定时器，设置为每2秒调用一次 capture_frame
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.capture_frame)  # 连接定时器触发信号到 capture_frame
        self.timer.start(5000)  # 每2秒触发一次

    def stop_timer(self):
        if self.timer is not None:
            self.timer.stop()  # 停止定时器
            self.timer = None

    def capture_frame(self):
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            self.label_video.setText("无法打开摄像头。")
            return

        ret, frame = self.capture.read()
        if not ret:
            return

        # 进行推理并处理图像
        data = APP.get_pic(frame, kd=1)  # 假设 `APP.get_pic()` 返回一个 DetectData 对象
        self.add_data(data)

        # 将图像转换为 RGB 格式并显示
        frame = APP.get_capFrame()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape
        bytes_per_line = channel * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.label_video.setPixmap(QPixmap.fromImage(q_image))
        if self.capture is not None:
            self.capture.release()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = UIWidgets()
    ex.show()
    sys.exit(app.exec_())
