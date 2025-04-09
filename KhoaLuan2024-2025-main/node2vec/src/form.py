# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import TextProcessor 
import nckh

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1430, 777)
        self.textEdit = QtWidgets.QTextEdit(Dialog)
        self.textEdit.setGeometry(QtCore.QRect(20, 60, 371, 311))
        self.textEdit.setMinimumSize(QtCore.QSize(371, 0))
        self.textEdit.setObjectName("textEdit")
        self.textEdit_2 = QtWidgets.QTextEdit(Dialog)
        self.textEdit_2.setGeometry(QtCore.QRect(440, 60, 371, 311))
        self.textEdit_2.setMinimumSize(QtCore.QSize(371, 0))
        self.textEdit_2.setObjectName("textEdit_2")
        self.textEdit_3 = QtWidgets.QTextEdit(Dialog)
        self.textEdit_3.setGeometry(QtCore.QRect(30, 530, 791, 231))
        self.textEdit_3.setObjectName("textEdit_3")
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(110, 400, 151, 28))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton.setFont(font)
        self.pushButton.setIconSize(QtCore.QSize(40, 40))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(Dialog)
        self.pushButton_2.setGeometry(QtCore.QRect(550, 390, 151, 28))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(Dialog)
        self.pushButton_3.setGeometry(QtCore.QRect(360, 480, 93, 28))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setObjectName("pushButton_3")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(20, 20, 151, 21))
        self.label.setMaximumSize(QtCore.QSize(151, 21))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setMidLineWidth(0)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(440, 20, 301, 31))
        self.label_2.setMaximumSize(QtCore.QSize(301, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(30, 470, 151, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.pushButton.setText(_translate("Dialog", "Thêm dữ liệu từ file"))
        self.pushButton_2.setText(_translate("Dialog", "Thêm dữ liệu từ file"))
        self.pushButton_3.setText(_translate("Dialog", "Tìm kiếm"))
        self.label.setText(_translate("Dialog", "Văn bản gốc"))
        self.label_2.setText(_translate("Dialog", "Nội dung cần tìm"))
        self.label_3.setText(_translate("Dialog", "Kết quả"))

class MainWindow(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        # Kết nối sự kiện
        self.pushButton.clicked.connect(self.add_data_from_file_1)
        self.pushButton_2.clicked.connect(self.add_data_from_file_2)
        self.pushButton_3.clicked.connect(self.search)

    def add_data_from_file_1(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Chọn file", "", "Text Files (*.txt)")
        if filename:
            with open(filename, "r", encoding="utf-8") as file:
                data = file.read()
                self.textEdit.setText(data)

    def add_data_from_file_2(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Chọn file", "", "Text Files (*.txt)")
        if filename:
            with open(filename, "r", encoding="utf-8") as file:
                data = file.read()
                self.textEdit_2.setText(data)
    #nut tim kiem o day
    def search(self):
        txt1 = self.textEdit.toPlainText().strip()  # Văn bản cần tìm
        txt2 = self.textEdit_2.toPlainText().strip()  # Văn bản nguồn chứa các câu

        # Kiểm tra đầu vào
        if not txt1 or not txt2:
            self.textEdit_3.setText("Vui lòng nhập đầy đủ văn bản!")
            return

        # Chia txt2 thành danh sách các câu
        lst = TextProcessor.split_sentences(txt2)
        if not lst:
            self.textEdit_3.setText("Không có câu nào trong văn bản nguồn!")
            return

        # Khởi tạo similarity checker (nên khởi tạo ở cấp lớp nếu dùng nhiều lần)
        try:
            similarity_checker = nckh.AStarTextSimilarity()
        except Exception as e:
            self.textEdit_3.setText(f"Lỗi khi khởi tạo AStarTextSimilarity: {e}")
            return

        # Tìm câu có độ tương đồng cao nhất
        best_match = 0.0
        ans = "Không có câu nào phù hợp"
        for sen in lst:
            sen = sen.strip()
            if not sen:  # Bỏ qua câu rỗng
                continue
            try:
                t = similarity_checker.calculate_similarity(txt1, sen)
                if t > best_match:
                    best_match = t
                    ans = sen
            except Exception as e:
                print(f"Lỗi khi tính độ tương đồng cho câu '{sen}': {e}")
                continue

        # Hiển thị kết quả
        self.textEdit_3.setText(f"{ans} (Độ tương đồng: {best_match:.4f})")

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
