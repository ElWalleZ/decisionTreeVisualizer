import subprocess
import sys
import os
import shutil
import numpy as np

from PyQt6 import QtWidgets, QtGui
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import QHBoxLayout, QMessageBox, QComboBox, QLabel, QPushButton
from treeMaker import TreeConverter
from model import entrenar_arbol_decision
from datasets import get_dataset_names, get_dataset, load_custom_dataset


class MiEtiqueta(QtWidgets.QScrollArea):
    """
    Widget personalizado para mostrar y navegar imágenes con scroll.

    Hereda de QScrollArea y contiene un QLabel para mostrar imágenes.
    Soporta arrastre del mouse para navegar por imágenes grandes.
    """
    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWidget(self.image_label)
        self.setStyleSheet("border: 1px solid black; background-color: white;")

    def set_image(self, pixmap):
        """
        Establece la imagen a mostrar en el widget.

        Args:
            pixmap (QPixmap): Imagen a mostrar.
        """
        self.image_label.setPixmap(pixmap)
        self.image_label.adjustSize()

        self.drag_start = None
        self.setMouseTracking(True)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_start = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drag_start is not None:
            delta = self.drag_start - event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() + delta.y())
            self.drag_start = event.pos()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.drag_start = None
        super().mouseReleaseEvent(event)

class Window(QtWidgets.QWidget):
    """
    Ventana principal de la aplicación de gestión de datasets y árboles de decisión.

    Proporciona interfaz para:
    - Cargar datasets (integrados o personalizados)
    - Entrenar modelos de árbol de decisión
    - Visualizar resultados y árboles de decisión
    - Gestionar proyectos
    """
    def __init__(self):
        super().__init__()
        self.current_dataset = None
        self._path = None
        self.current_project = "Project"
        self.project_dir = None
        self.current_dataset_name = ""
        self.modelo = None
        self.center()

        self.init_ui()

    def center(self):
        qr = self.frameGeometry()
        cp = self.screen().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def init_ui(self):
        self.viewer = QtWidgets.QTextEdit()
        self.viewer.setReadOnly(True)
        self.viewer.setStyleSheet("border: 1px solid black; font-size: 14px;")
        self.viewer.setLineWrapMode(QtWidgets.QTextEdit.LineWrapMode.WidgetWidth)
        self.viewer.setWordWrapMode(QtGui.QTextOption.WrapMode.WordWrap)

        self.viewer2 = MiEtiqueta()

        self.lbl_project = QtWidgets.QLabel("Proyecto: Sin nombre")

        self.btn_entrenar = QtWidgets.QPushButton("Entrenar Árbol de Decisión")
        self.btn_entrenar.clicked.connect(self.entrenar_modelo)
        self.btn_entrenar.setMinimumSize(QSize(200, 50))

        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(['Seleccionar dataset'] + get_dataset_names())
        self.dataset_combo.currentIndexChanged.connect(self.load_selected_dataset)
        self.dataset_combo.setMinimumSize(QSize(200, 50))

        self.btn_custom_dataset = QtWidgets.QPushButton("Cargar dataset personalizado")
        self.btn_custom_dataset.clicked.connect(self.load_custom_dataset)
        self.btn_custom_dataset.setMinimumSize(QSize(200, 50))

        self.depthValue = QtWidgets.QSpinBox()
        self.depthValue.setValue(5)
        self.depthValue.setMinimum(1)
        self.depthValue.setMaximum(10)
        self.depthValue.setFixedSize(70,40)

        self.btn_new_project = QPushButton("Nuevo Proyecto")
        self.btn_new_project.clicked.connect(self.nuevo_proyecto)
        self.btn_new_project.setMinimumSize(QSize(200, 50))

        layout = QtWidgets.QGridLayout(self)

        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 2)

        dataset_layout = QHBoxLayout()
        dataset_layout.addWidget(self.dataset_combo)
        dataset_layout.addWidget(self.btn_custom_dataset)

        button_layout = QHBoxLayout()
        label = QLabel("Profundidad:")
        label.setFixedSize(100,20)

        button_layout.addWidget(label)
        button_layout.addWidget(self.depthValue)
        button_layout.addWidget(self.btn_new_project)
        button_layout.addWidget(self.btn_entrenar)

        layout.addLayout(dataset_layout, 0, 0, 1, 2)  # Ocupa ambas columnas
        layout.addLayout(button_layout, 1, 0, 1, 2)  # Ocupa ambas columnas
        layout.addWidget(self.viewer, 2, 0)  # Columna izquierda
        layout.addWidget(self.viewer2, 2, 1)  # Columna derecha

        self.viewer.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Expanding
        )
        self.viewer2.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding
        )

        self.viewer.setMinimumSize(440, 680)
        self.viewer2.setMinimumSize(840, 680)

        self.setMinimumSize(1800, 840)
        self.setWindowTitle("Dataset Manager")

    def rename_project(self):
        """Solicita nuevo nombre para el proyecto actual."""
        new_name, ok = QtWidgets.QInputDialog.getText(
            self,
            "Nombre del Proyecto",
            "Ingrese el nombre del proyecto:",
            text=self.current_project
        )

        if ok and new_name:
            self.current_project = new_name.strip()
            self.lbl_project.setText(f"Proyecto: {self.current_project}")
            self.create_project_directory()

    def create_project_directory(self):
        """Crea directorio para el proyecto actual si no existe."""
        if self.current_project and self.current_project != "Project":
            self.project_dir = os.path.join(os.getcwd(), "proyectos", self.current_project)
            os.makedirs(self.project_dir, exist_ok=True)
        else:
            self.project_dir = None

    def load_selected_dataset(self, index):
        """
        Carga el dataset seleccionado en el combobox.

        Args:
            index (int): Índice del item seleccionado en el combobox.
        """
        if index > 0:
            if self.current_project == "Project":
                self.rename_project()
                if self.current_project == "Project":
                    self.dataset_combo.setCurrentIndex(0)
                    return

            dataset_name = self.dataset_combo.itemText(index)
            if dataset := get_dataset(dataset_name):
                self.current_dataset_name = dataset_name
                self.current_dataset = dataset
                self.show_dataset_info(dataset)
                QMessageBox.information(self, "Dataset cargado",
                                        f"Dataset {dataset_name} cargado exitosamente!")
            self.show_dataset_info(dataset)
            self.save_dataset_to_project()

    def save_dataset_to_project(self):
        """Guarda el dataset actua (si no pertenece a los predeterminados) en el directorio del proyecto."""
        if self.current_dataset and self.project_dir:
            try:
                if hasattr(self.current_dataset, 'filename'):
                    if os.path.isfile(self.current_dataset.filename):
                        shutil.copy(self.current_dataset.filename, self.project_dir)
                    else:
                        QMessageBox.warning(self, "Advertencia",
                                            f"Archivo original no encontrado: {self.current_dataset.filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error",
                                     f"Error guardando proyecto:\n{str(e)}")

    def load_custom_dataset(self):
        """
        Carga el dataset seleccionado en el combobox.

        Args:
            index (int): Índice del item seleccionado en el combobox.
        """
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Cargar dataset personalizado", "", "CSV Files (*.csv)")

        if path:
            try:
                if self.current_project == "Project":
                    self.rename_project()
                    if self.current_project == "Project":
                        self.dataset_combo.setCurrentIndex(0)
                        return

                self.current_dataset = load_custom_dataset(path)
                self.current_dataset_name = os.path.splitext(os.path.basename(path))[0]
                self.show_dataset_info(self.current_dataset)
                self.save_dataset_to_project()
                QMessageBox.information(self, "Éxito", "Dataset personalizado cargado!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error cargando dataset: {str(e)}")

    def show_dataset_info(self, dataset):
        """
        Muestra información detallada del dataset cargado.

        Args:
            dataset (Bunch): Dataset a mostrar.
        """
        safe_get = lambda attr, default: getattr(dataset, attr, default) if hasattr(dataset, attr) else default

        info = f"""
        === Proyecto: {self.current_project} ===

        Dataset: {self.current_dataset_name}
        Tipo: {safe_get('dataset_type', 'Propio')}

        Muestras: {dataset.data.shape[0]}
        Características: {dataset.data.shape[1]}

        Target:
        - Clases: {len(np.unique(dataset.target))}
        - Nombres: {', '.join(map(str, safe_get('target_names', [])))}

        Archivo: {safe_get('filename', 'Dataset integrado')}
        """
        info += f"\nAcciones disponibles:\n- Clases: {len(np.unique(dataset.target))}"

        self.viewer.setPlainText(info)

    def closeEvent(self, event):
        if self.current_project != "Sin nombre":
            self.save_dataset_to_project()
        super().closeEvent(event)

    def entrenar_modelo(self):
        """
        Entrena un modelo de árbol de decisión con el dataset actual.

        Realiza:
        - Entrenamiento del modelo
        - Generación de visualización
        - Conversión a PDF/PNG
        - Muestra resultados
        """
        if not self.current_dataset:
            QMessageBox.warning(self, "Error", "Primero selecciona un dataset")
            return

        if self.current_project == "Project":
            QMessageBox.warning(self, "Error", "Debes asignar un nombre al proyecto antes de entrenar")
            self.rename_project()
            if self.current_project == "Project":
                return

        try:
            resultado = entrenar_arbol_decision(self.current_dataset, max_depth= self.depthValue.value())

            texto_limpio = resultado['texto_arbol'].replace('_', '')
            mensaje = f"""=== Resultados del Modelo ===
                Tipo: {resultado['tipo'].capitalize()}
                {resultado['metric_name']}: {resultado['metrica']:.2%}

                === Estructura del Árbol ===
                {texto_limpio}
                """
            self.viewer.setPlainText(mensaje)

            output_dir = os.path.join(self.project_dir, "output") if self.project_dir else "output"
            os.makedirs(output_dir, exist_ok=True)

            pdf_path = TreeConverter.convert_to_pdf( #convertir a PDF
                texto_limpio,
                output_dir=output_dir,
                filename="decision_tree"
            )

            img_path = pdf_path.replace('.pdf', '.png') #PDF a imagen
            try:
                subprocess.run([
                    "pdftoppm",
                    "-png",
                    "-singlefile",
                    pdf_path,
                    os.path.join(output_dir, "temp_tree")
                ], check=True, capture_output=True)

                temp_img = os.path.join(output_dir, "temp_tree.png")
                if os.path.exists(temp_img):
                    os.rename(temp_img, img_path)

                if not os.path.exists(img_path):
                    subprocess.run([
                        "convert",
                        "-density", "300",
                        pdf_path,
                        "-quality", "90",
                        img_path
                    ], check=True)

            except subprocess.CalledProcessError as e:
                error_msg = f"Error al convertir PDF a imagen:\n{e.stderr.decode()}"
                if "pdftoppm" in error_msg:
                    error_msg += "\n\nInstala poppler-utils con:\nsudo apt-get install poppler-utils"
                elif "convert" in error_msg:
                    error_msg += "\n\nInstala ImageMagick con:\nsudo apt-get install imagemagick"
                QMessageBox.warning(self, "Advertencia", error_msg)
                return

            if self.project_dir:
                with open(os.path.join(self.project_dir, 'arbol.txt'), 'w') as f:
                    f.write(texto_limpio)

            self.mostrar_imagen_arbol(img_path)
            QMessageBox.information(self, "Éxito", "Modelo entrenado y visualizado!")

        except Exception as e:
            error_msg = str(e)
            if "LaTeX" in error_msg:
                error_msg += "\n\nAsegúrate de tener instalados:\nsudo apt-get install texlive-full texlive-latex-extra poppler-utils"
            QMessageBox.critical(self, "Error", error_msg)

    def mostrar_imagen_arbol(self, img_path):
        """
        Muestra la imagen del árbol de decisión en el visor.

        Args:
            img_path (str): Ruta al archivo de imagen.
        """
        if not img_path or not os.path.exists(img_path):
            self.viewer2.clear()
            return

        try:
            pixmap = QtGui.QPixmap(img_path)
            if pixmap.isNull():
                self.viewer2.clear()
                return

            self.viewer2.set_image(pixmap)

        except Exception as e:
            print(f"Error al mostrar imagen: {str(e)}")
            self.viewer2.clear()

    def nuevo_proyecto(self):
        """Crea un nuevo proyecto, reiniciando el estado de la aplicación."""
        new_name, ok = QtWidgets.QInputDialog.getText(
            self,
            "Nuevo Proyecto",
            "Ingrese el nombre del nuevo proyecto:",
        )

        if ok and new_name:
            self.current_project = new_name.strip()
            self.current_dataset = None
            self.current_dataset_name = ""
            self.modelo = None

            self.lbl_project.setText(f"Proyecto: {self.current_project}")
            self.dataset_combo.setCurrentIndex(0)
            self.viewer.clear()
            self.viewer2.clear()

            self.create_project_directory()

            QMessageBox.information(
                self,
                "Nuevo Proyecto",
                f"Proyecto '{self.current_project}' creado!"
            )

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec())