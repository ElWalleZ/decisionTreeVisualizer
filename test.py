import sys
import os
import subprocess
import numpy as np
import cv2
from PyQt6 import QtWidgets, QtGui
from PyQt6.QtCore import pyqtSignal, QSize
from PyQt6.QtWidgets import QHBoxLayout, QMessageBox

"""
Nomás como comentario así extra, nuño pide modulación en los códigos, vaya en diferentes archivos, 
por si lo quieres tomar en cuenta, es que es un requisito que viene en las diapos, espero esta base te ayude, buenas noches xd
"""

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

        self.procesarImagenEntrada = QtWidgets.QPushButton("Procesar")
        self.procesarImagenEntrada.setMinimumSize(BUTTON_SIZE)
        self.procesarImagenEntrada.clicked.connect(self.ProcesarImage)

        self.guardarImagen = QtWidgets.QPushButton("Guardar Imagen")
        self.guardarImagen.setMinimumSize(BUTTON_SIZE)
        self.guardarImagen.clicked.connect(self.handleSaveFile)

        self.botonProcesaReservado = QtWidgets.QPushButton("Guardar Árbol Texto")
        self.botonProcesaReservado.setMinimumSize(BUTTON_SIZE)
        self.botonProcesaReservado.clicked.connect(self.guardarTextoArbol)

        self.generarLatexBtn = QtWidgets.QPushButton("Generar Árbol LaTeX")
        self.generarLatexBtn.setMinimumSize(BUTTON_SIZE)
        self.generarLatexBtn.clicked.connect(self.generar_arbol_latex)

        layout = QtWidgets.QGridLayout(self)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.buttonOpen)
        button_layout.addWidget(self.botonProcesaReservado)
        button_layout.addWidget(self.generarLatexBtn)
        button_layout.addWidget(self.guardarImagen)

        layout.addLayout(button_layout, 0, 0, 1, 4)
        layout.addWidget(self.viewer, 1, 0, 1, 2)
        layout.addWidget(self.viewer2, 1, 2, 1, 2)

    def ProcesarImage(self):
        pass

    def handleSaveFile(self):
        if self.OpenCV_image2 is not None:
            defaultname = "example.png"
            fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Guardar archivo", defaultname, "Images(*.jpg *.png)")
            if fileName:
                if not fileName.endswith(('.png', '.jpg')):
                    fileName += ".png"
                cv2.imwrite(fileName, self.OpenCV_image2)
        else:
            QMessageBox.warning(self, "Error", "No hay nada que guardar aún.")

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

    # Nuevas Funciones, ola

    def texto_a_latex(self, texto):
        def convertir_lineas_a_qtree(lineas):
            stack = []
            output = ""
            for linea in lineas:
                espacios = len(linea) - len(linea.lstrip())
                nivel = espacios // 4
                texto = linea.strip()
                while len(stack) > nivel:
                    output += "]"
                    stack.pop()
                output += f"[.{texto} "
                stack.append(texto)
            output += "]" * len(stack)
            return output

        contenido_qtree = convertir_lineas_a_qtree(texto.strip().split("\n"))

        latex = r"""\documentclass{article}
        \usepackage[utf8]{inputenc}
        \usepackage{qtree}
        \usepackage[paperwidth=15cm, paperheight=15cm, margin=1cm]{geometry}
        \pagestyle{empty}
        \begin{document}
        \centering
        \Tree """ + contenido_qtree + "\n\\end{document}"

        return latex

    def generar_pdf_y_imagen(self, tex_code, output_dir="output_arbol", nombre="arbol"):
        os.makedirs(output_dir, exist_ok=True)
        tex_path = os.path.join(output_dir, f"{nombre}.tex")
        pdf_path = os.path.join(output_dir, f"{nombre}.pdf")
        img_path = os.path.join(output_dir, f"{nombre}.png")

        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(tex_code)

        # Ejecutar pdflatex y capturar salida
        try:
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", f"{nombre}.tex"],  # <-- SOLO EL NOMBRE DEL ARCHIVO
                cwd=output_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
            )

            if result.returncode != 0 or not os.path.exists(pdf_path):
                log = result.stdout.decode(errors="ignore")
                QMessageBox.critical(self, "Error de LaTeX", f"No se pudo compilar el archivo .tex:\n\n{log[:1000]}")
                return None
        except FileNotFoundError:
            QMessageBox.critical(self, "Error", "No se encontró 'pdflatex'.")
            return None
        except subprocess.TimeoutExpired:
            QMessageBox.critical(self, "Error", "La compilación excedió el tiempo límite.")
            return None

        # Convertir PDF a imagen
        try:
            subprocess.run(
                ["pdftoppm", pdf_path, os.path.join(output_dir, nombre), "-png", "-singlefile"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10
            )
        except FileNotFoundError:
            QMessageBox.critical(self, "Error", "No se encontró 'pdftoppm'.")
            return None

        return img_path if os.path.exists(img_path) else None



    def generar_arbol_latex(self):
        texto = self.viewer.toPlainText()
        if not texto.strip():
            QMessageBox.warning(self, "Advertencia", "No hay texto para procesar.")
            return

        latex_code = self.texto_a_latex(texto)
        img_path = self.generar_pdf_y_imagen(latex_code)

        if img_path and os.path.exists(img_path):
            image = cv2.imread(img_path)
            self.OpenCV_image2 = image
            self.ActualizarPixMap2(image)
            QMessageBox.information(self, "Éxito", f"Árbol generado y guardado en {img_path}")
        else:
            QMessageBox.critical(self, "Error", "Error al generar la imagen del árbol.")


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.setWindowTitle("Decision Tree Visualizer")
    window.show()
    sys.exit(app.exec())

