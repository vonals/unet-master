# 界面读入文件
#
import cv2
import sys

from IPython.core.inputtransformer2 import tr
from PyQt5 import QtWidgets
from PySide2.QtWidgets import QApplication, QMainWindow, QSpinBox, QPushButton, \
    QPlainTextEdit, QMessageBox, QLabel, QFileDialog, QHBoxLayout
from PySide2.QtCore import Slot
from PySide2.QtUiTools import QUiLoader
from PySide2.QtGui import QIcon, QPixmap, QImage, QPainter
# import pyqtgraph as pg
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
from model import *

class Stats:

    def __init__(self):

        # 从文件中加载UI定义
        # 从 UI 定义中动态 创建一个相应的窗口对象
        # 注意：里面的控件对象也成为窗口对象的属性了
        # 比如 self.ui.button , self.ui.textEdit
        loader = QUiLoader()
        # loader.registerCustomWidget(QImage)
        self.ui = QUiLoader().load('UI/main.ui')
        # 菜单栏
        self.ui.actionopen.triggered.connect(self.handleAddFile)
        # 按钮
        self.ui.pushButton.clicked.connect(self.handleShowPic)

        # self.pix = QPixmap(300, 300)  # 默认填充颜色为黑色
        # self.pix.fill('red')  # 修改填充颜色为红色
        # self.ui.IMG.setPixmap(self.pix)  # 设定 QLabel 的 pixmap


    def handleShowPic(self):

        self.ui.textEdit.setText("AAAAA")
    # 打开文件处理
    def handleAddFile(self):
        # self.ui.textEdit.setText("AAAAA")
        # 打开文件（一定要写self.ui），“框名”，“打开的文件目录”，“可打开的文件类型”
        filename,imgType  = QFileDialog.getOpenFileName(self.ui,
                                               "Open Image",
                                               "./data/test",
                                               "Image Files (*.png *.jpg *.bmp *.tif)")

        # 初始化 QPixmap 类
        img = QPixmap(filename)
        # 设定 QLabel 的 pixmap
        self.ui.label_2.setPixmap(img)
        # # 加载预训练网络
        # model = unet("unet_membrane.hdf5")
        # img2 = load_img(filename)
        # img2 = img_to_array(img2)
        # img2 = np.expand_dims(img2, axis=0)
        # result = model.predict(img2)
        # self.ui.label.setPixmap(result)
        print(img)
        print('FileDirectory')


app = QApplication([])
# 加载 icon
app.setWindowIcon(QIcon('logo.png'))
stats = Stats()
# 显示ui
stats.ui.show()


# 关闭窗口
app.exec_()