from PySide2.QtWidgets import QApplication, QMainWindow, QSpinBox, QPushButton,  QPlainTextEdit

from PySide2.QtCore import Slot

@Slot()
def handleCalc():
    print('finish!')

app = QApplication([])

Window = QMainWindow()
Window.resize(350, 300)
Window.setWindowTitle('肺实质分割')
textEdit = QPlainTextEdit(Window)
textEdit.setPlaceholderText("")

button = QPushButton('分割',Window)
button.move(120,0)
button.clicked.connect(handleCalc)
Window.show()
app.exec_()