import sys
import numpy as np
import cv2
from PyQt6 import QtWidgets, QtGui
from PyQt6.QtCore import pyqtSignal, QSize
from PyQt6.QtWidgets import QHBoxLayout, QMessageBox


class MiEtiqueta(QtWidgets.QLabel):
    def __init__(self):
        super().__init__()
        self.Lista = []
        self.setStyleSheet("border: 1px solid black;")

class Window(QtWidgets.QWidget):

    def center(self):
        """
        Centra la Ventada SI o SI
        """
        qr = self.frameGeometry()
        cp = self.screen().availableGeometry().center()

        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def __init__(self):
        super().__init__()
        self.OpenCV_image3 = None
        self.OpenCV_image = None
        self.OpenCV_image2 = None
        self.center()

        self._path = None

        self.viewer = QtWidgets.QTextEdit()
        self.viewer.setReadOnly(False)
        self.viewer.setStyleSheet("border: 1px solid black; font-size: 14px;")
        self.viewer.setFixedSize(840, 680)

        self.viewer2 = MiEtiqueta()
        self.viewer2.setFixedSize(840, 680)
        self.viewer2.setScaledContents(True)

        self.buttonOpen = QtWidgets.QPushButton("Cargar Árbol (.txt)")
        BUTTON_SIZE = QSize(200, 50)
        self.buttonOpen.setMinimumSize(BUTTON_SIZE)
        self.buttonOpen.clicked.connect(self.handleOpen)

        self.elements = []

        self.procesarImagenEntrada = QtWidgets.QPushButton("Procesar")
        self.procesarImagenEntrada.setMinimumSize(BUTTON_SIZE)
        self.procesarImagenEntrada.clicked.connect(self.ProcesarImage)

        self.guardarImagen = QtWidgets.QPushButton("Guardar")
        self.guardarImagen.setMinimumSize(BUTTON_SIZE)
        self.guardarImagen.clicked.connect(self.handleSaveFile)

        layout = QtWidgets.QGridLayout(self)
        self.botonProcesaReservado = QtWidgets.QPushButton("Buscar Señales de Trafico")
        self.botonProcesaReservado.setMinimumSize(BUTTON_SIZE)
        self.botonProcesaReservado.clicked.connect(self.guardarTextoArbol)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.buttonOpen)
        #button_layout.addWidget(self.procesarImagenEntrada)
        button_layout.addWidget(self.botonProcesaReservado)
        button_layout.addWidget(self.guardarImagen)

        layout.addLayout(button_layout, 0, 0, 1, 4)
        layout.addWidget(self.viewer, 1, 0, 1, 2)
        layout.addWidget(self.viewer2, 1, 2, 1, 2)

        Tamano = (self.viewer.size().width(), self.viewer.size().height())

        print(self.viewer.size(), type(self.viewer.size()), Tamano)

    def ProcesarImage(self):
        pass

    def handleSaveFile(self):
        if self.OpenCV_image2 is not None:
            defaultname = "example.png"

            fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save File", defaultname,
                                                                "Images(*.jpg *.png)")

            if fileName:
                if not fileName.endswith(('.png', '.jpg')):
                    fileName += ".png"
                cv2.imwrite(fileName, self.OpenCV_image2)
        else:
            QMessageBox.warning(self, "Error", "No hay nada que guardar aun")

    def handleOpen(self):
        path = QtWidgets.QFileDialog.getOpenFileName(self, "Selecciona archivo de árbol", ".", "Text Files (*.txt)")[0]
        if path:
            self._path = path
            with open(self._path, 'r', encoding='utf-8') as f:
                contenido = f.read()
            self.viewer.setPlainText(contenido)
        else:
            QMessageBox.warning(self, "Error", "No se seleccionó ningún archivo válido")

    def obtener_texto_arbol(self):
        return self.viewer.toPlainText()

    def guardarTextoArbol(self):
        texto = self.viewer.toPlainText()

        if not texto.strip():
            QMessageBox.warning(self, "Advertencia", "No hay texto para guardar.")
            return

        if self._path:  # Ya hay archivo cargado, sobrescribimos
            try:
                with open(self._path, 'w', encoding='utf-8') as f:
                    f.write(texto)
                QMessageBox.information(self, "Éxito", f"Texto guardado en: {self._path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"No se pudo guardar el archivo:\n{e}")
        else:
            # No hay archivo cargado, pedimos ruta para guardar
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Guardar archivo como", ".", "Text Files (*.txt)")
            if path:
                if not path.endswith(".txt"):
                    path += ".txt"
                try:
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(texto)
                    self._path = path
                    QMessageBox.information(self, "Éxito", f"Archivo creado y texto guardado en:\n{path}")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"No se pudo guardar el archivo:\n{e}")
            else:
                QMessageBox.information(self, "Cancelado", "No se creó ningún archivo.")


    def ActualizarPixMap(self):
        display_width = self.viewer.width()
        display_height = self.viewer.height()
        resized = cv2.resize(self.OpenCV_image3, (display_width, display_height), interpolation=cv2.INTER_LINEAR)
        QImageTemp = QtGui.QImage(
            cv2.cvtColor(resized, cv2.COLOR_BGR2RGB),
            resized.shape[1],
            resized.shape[0],
            resized.shape[1] * 3,
            QtGui.QImage.Format.Format_RGB888
        )
        self.viewer.setPixmap(QtGui.QPixmap(QImageTemp))

    def ActualizarPixMap2(self, image):
        display_width = self.viewer2.width()
        display_height = self.viewer2.height()
        resized_image = cv2.resize(image, (display_width, display_height), interpolation=cv2.INTER_LINEAR)
        qimage = QtGui.QImage(
            cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB),
            resized_image.shape[1],
            resized_image.shape[0],
            resized_image.shape[1] * 3,
            QtGui.QImage.Format.Format_RGB888
        )
        self.viewer2.setPixmap(QtGui.QPixmap(qimage))

    # def ActualizarImagen(self):
    #     self.OpenCV_image = cv2.imread(self._path)
    #     self.OpenCV_image3 = self.OpenCV_image.copy()
    #     displaysize = (self.viewer.width(), self.viewer.height())
    #     self.OpenCV_image3 = cv2.resize(self.OpenCV_image3, displaysize, interpolation=cv2.INTER_LINEAR)
    #     QImageTemp = QtGui.QImage(
    #         cv2.cvtColor(self.OpenCV_image3, cv2.COLOR_BGR2RGB),
    #         self.OpenCV_image3.shape[1],
    #         self.OpenCV_image3.shape[0],
    #         self.OpenCV_image3.shape[1] * 3,
    #         QtGui.QImage.Format.Format_RGB888
    #     )
    #     pixmap = QtGui.QPixmap(QImageTemp)
    #     self.viewer.setPixmap(pixmap)
    #     self.viewer2.setPixmap(pixmap)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.setWindowTitle("Traffic Sign Detector")
    window.show()
    sys.exit(app.exec())