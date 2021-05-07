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
import matplotlib.pyplot as plt
from model import *
from NewUNet import *
from data import *

class Stats:

    def __init__(self):

        # 从文件中加载UI定义
        # 从 UI 定义中动态 创建一个相应的窗口对象
        # 注意：里面的控件对象也成为窗口对象的属性了
        # 比如 self.ui.button , self.ui.textEdit

        # loader = QUiLoader()
        # loader.registerCustomWidget(QImage)
        self.ui = QUiLoader().load('UI/main.ui')
        # 菜单栏
        self.ui.actionopen.triggered.connect(self.handleAddFile)
        # 按钮
        self.ui.pushButton.clicked.connect(self.handleShowPic)



    def handleShowPic(self):
        a=5
        # self.ui.textEdit.setText("AAAAA")

    # 打开文件处理
    def handleAddFile(self,model):
        # 打开文件（一定要写self.ui），“框名”，“打开的文件目录”，“可打开的文件类型”
        filename, imgType = QFileDialog.getOpenFileName(self.ui,
                                               "Open Image",
                                               "./data/test",
                                               "Image Files (*.png *.jpg *.bmp *.tif)")
        self.ui.textEdit.setText("成功打开"+filename)
        # 初始化 QPixmap 类
        img = QPixmap(filename)
        # 设定 QLabel 的 pixmap
        self.ui.label_2.setPixmap(img)


        # # 加载预训练网络
        model = UNet1("unet_membrane.hdf5")
        qGene = QGenerator(filename)
        result = model.predict_generator(qGene, 1, verbose=1)
        saveResult("data/membrane/temp", result)
        tempfile="data/membrane/temp/0_predict.png"
        # 初始化 QPixmap 类
        img2 = QPixmap(tempfile)
        img2 = img2.scaled(512,512)
        # 设定 QLabel 的 pixmap
        self.ui.label.setPixmap(img2)
        # plt.imshow(result)
        # plt.show()

        print(img)
        print(img2)
        print('FileDirectory')


app = QApplication([])
# 加载 icon
app.setWindowIcon(QIcon('logo.png'))
stats = Stats()
# 显示ui
stats.ui.show()
# 关闭窗口
app.exec_()