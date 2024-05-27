import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
import torch
import cv2
from webcam_pyuic import Ui_MainWindow  # 변환된 UI 파일 임포트

class ObjectDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle('YOLOv5 Object Detection')
        self.setGeometry(100, 100, 1200, 600)  # 가로 크기를 넓힘
        self.initUI()
        self.model = torch.hub.load('./yolov5', model='custom', path='./pt/yolov5l.pt', source='local')
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.detect_webcam)
        self.webcam = cv2.VideoCapture(0)  # 웹캠 열기, 여러 대의 카메라가 있는 경우 인덱스 지정 가능

    def initUI(self):
        self.ui.start_webcam_button.clicked.connect(self.start_webcam)

    def start_webcam(self):
        if not self.timer.isActive():
            self.timer.start(10)  # 10ms마다 타이머 이벤트 발생

    def detect_webcam(self):
        ret, frame = self.webcam.read()  # 프레임 캡처
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 원본 이미지를 화면에 표시
            self.display_image(frame_rgb, self.ui.image_label)

            # 모델 실행 및 결과 표시
            results = self.model(frame)
            results.render()  # Draw boxes and labels on the image
            rendered_img = results.ims[0]
            rendered_img_rgb = cv2.cvtColor(rendered_img, cv2.COLOR_BGR2RGB)
            self.display_image(rendered_img_rgb, self.ui.result_label)

    def display_image(self, image, label):
        if image is None or image.size == 0:
            return
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        label.setPixmap(pixmap)
        label.setScaledContents(True)
        label.setText("")

def main():
    app = QApplication(sys.argv)
    window = ObjectDetectionApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
