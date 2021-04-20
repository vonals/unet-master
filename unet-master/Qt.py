from PySide2.QtWidgets import QApplication, QMainWindow, QSpinBox
from PySide2.QtCore import Slot

app = QApplication([])

MainWindow = QMainWindow()

SpinBox = QSpinBox(MainWindow)
SpinBox.resize(100, 20)
SpinBox.value()
SpinBox.setRange(0, 100)                        #设置数值范围

MainWindow.show()
app.exec_()