import subprocess
import sys
import os
from datetime import datetime
import json
import shutil

import numpy as np
from PyQt6 import QtWidgets, QtGui
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import QHBoxLayout, QMessageBox, QComboBox, QLabel, QPushButton
from decision_tree_git_que_puso_el_profe import TreeConverter

from model import entrenar_arbol_decision
from datasets import get_dataset_names, get_dataset, load_custom_dataset


class MiEtiqueta(QtWidgets.QLabel):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("border: 1px solid black;")


class Window(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.current_dataset = None
        self._path = None
        self.current_project = "Sin nombre"
        self.project_dir = None
        self.current_dataset_name = ""
        self.modelo = None
        self.center()

        # Configuración inicial de la interfaz
        self.init_ui()

    def center(self):
        qr = self.frameGeometry()
        cp = self.screen().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def init_ui(self):
        # Widget principal para mostrar información
        self.viewer = QtWidgets.QTextEdit()
        self.viewer.setReadOnly(True)
        self.viewer.setStyleSheet("border: 1px solid black; font-size: 14px;")
        self.viewer.setFixedSize(840, 680)

        # Widget para mostrar imágenes (no usado actualmente)
        self.viewer2 = MiEtiqueta()
        self.viewer2.setFixedSize(840, 680)
        self.viewer2.setScaledContents(True)

        self.lbl_project = QtWidgets.QLabel("Proyecto: Sin nombre")
        self.btn_rename_project = QtWidgets.QPushButton("Cambiar nombre")
        self.btn_rename_project.clicked.connect(self.rename_project)

        self.btn_entrenar = QtWidgets.QPushButton("Entrenar Árbol de Decisión")
        self.btn_entrenar.clicked.connect(self.entrenar_modelo)
        self.btn_entrenar.setMinimumSize(QSize(200, 50))

        project_layout = QHBoxLayout()
        project_layout.addWidget(self.lbl_project)
        project_layout.addWidget(self.btn_rename_project)

        # Controles para datasets
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(['Seleccionar dataset'] + get_dataset_names())
        self.dataset_combo.currentIndexChanged.connect(self.load_selected_dataset)
        self.dataset_combo.setMinimumSize(QSize(200, 50))


        self.btn_custom_dataset = QtWidgets.QPushButton("Cargar dataset personalizado")
        self.btn_custom_dataset.clicked.connect(self.load_custom_dataset)
        self.btn_custom_dataset.setMinimumSize(QSize(200, 50))


        #self.buttonOpen = QtWidgets.QPushButton("Abrir archivo")
        #self.buttonOpen.setMinimumSize(QSize(200, 50))

        self.depthValue = QtWidgets.QSpinBox()
        self.depthValue.setValue(5)
        self.depthValue.setMinimum(1)
        self.depthValue.setMaximum(10)
        self.depthValue.setFixedSize(70,40)

        self.btn_new_project = QPushButton("Nuevo Proyecto")
        self.btn_new_project.clicked.connect(self.nuevo_proyecto)
        self.btn_new_project.setMinimumSize(QSize(200, 50))


        # Layout principal
        layout = QtWidgets.QGridLayout(self)

        # Sección de datasets
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

        # Ensamblado final
        layout.addLayout(project_layout, 0, 0, 1, 4)
        layout.addLayout(dataset_layout, 0, 0, 1, 4)
        layout.addLayout(button_layout, 1, 0, 1, 4)
        layout.addWidget(self.viewer, 2, 0, 1, 1)
        layout.addWidget(self.viewer2, 2, 2, 1, 1)


        self.setWindowTitle("Dataset Manager")
        self.setMinimumSize(1200, 800)

    def rename_project(self):
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
        if self.current_project and self.current_project != "Sin nombre":
            self.project_dir = os.path.join(os.getcwd(), "proyectos", self.current_project)
            os.makedirs(self.project_dir, exist_ok=True)
        else:
            self.project_dir = None

    def load_selected_dataset(self, index):
        if index > 0:
            self.current_dataset_name = self.dataset_combo.currentText()
            if self.current_project == "Sin nombre":
                self.rename_project()
                if self.current_project == "Sin nombre":  # Si canceló
                    return

            dataset_name = self.dataset_combo.itemText(index)
            if dataset := get_dataset(dataset_name):
                self.current_dataset = dataset
                self.show_dataset_info(dataset)
                QMessageBox.information(self, "Dataset cargado",
                                        f"Dataset {dataset_name} cargado exitosamente!")
            self.show_dataset_info(dataset)
            self.save_dataset_to_project()

    def save_dataset_to_project(self):
        if self.current_dataset and self.project_dir:
            try:
                # Crear metadata segura
                metadata = {
                    'proyecto': self.current_project,
                    'dataset_type': 'scikit-learn',
                    'dataset_name': self.current_dataset_name,
                    'fecha': datetime.now().isoformat(),
                    'features': self.current_dataset.data.shape[1],
                    'samples': self.current_dataset.data.shape[0]
                }

                # Si es dataset personalizado, añadir info extra
                if hasattr(self.current_dataset, 'filename'):
                    metadata.update({
                        'dataset_type': 'custom',
                        'original_file': os.path.basename(self.current_dataset.filename)
                    })

                    # Verificar existencia real del archivo antes de copiar
                    if os.path.isfile(self.current_dataset.filename):
                        shutil.copy(self.current_dataset.filename, self.project_dir)
                    else:
                        QMessageBox.warning(self, "Advertencia",
                                            f"Archivo original no encontrado: {self.current_dataset.filename}")

                # Guardar metadata siempre
                with open(os.path.join(self.project_dir, 'metadata.json'), 'w') as f:
                    json.dump(metadata, f, indent=4)

            except Exception as e:
                QMessageBox.critical(self, "Error",
                                     f"Error guardando proyecto:\n{str(e)}")

    def load_custom_dataset(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Cargar dataset personalizado", "", "CSV Files (*.csv)")

        if path:
            try:
                self.current_dataset = load_custom_dataset(path)
                self.current_dataset_name = os.path.splitext(os.path.basename(path))[0]
                self.show_dataset_info(self.current_dataset)
                self.save_dataset_to_project()
                QMessageBox.information(self, "Éxito", "Dataset personalizado cargado!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error cargando dataset: {str(e)}")

    def show_dataset_info(self, dataset):
        # Obtener información de manera segura
        safe_get = lambda attr, default: getattr(dataset, attr, default) if hasattr(dataset, attr) else default

        info = f"""
        === Proyecto: {self.current_project} ===

        Dataset: {self.current_dataset_name}
        Tipo: {safe_get('dataset_type', 'scikit-learn')}

        Muestras: {dataset.data.shape[0]}
        Características: {dataset.data.shape[1]}

        Target:
        - Clases: {len(np.unique(dataset.target))}
        - Nombres: {', '.join(map(str, safe_get('target_names', [])))}

        Archivo: {safe_get('filename', 'Dataset integrado')}
        """
        info += f"\nAcciones disponibles:\n- Clases: {len(np.unique(dataset.target))} (apto para clasificación)"

        self.viewer.setPlainText(info)

    def closeEvent(self, event):
        # Guardar automáticamente al cerrar
        if self.current_project != "Sin nombre":
            self.save_dataset_to_project()
        super().closeEvent(event)

    def entrenar_modelo(self):
        if not self.current_dataset:
            QMessageBox.warning(self, "Error", "Primero selecciona un dataset")
            return

        try:
            # 1. Entrenar modelo
            resultado = entrenar_arbol_decision(self.current_dataset, max_depth= self.depthValue.value())

            # 2. Mostrar texto del árbol
            texto_limpio = resultado['texto_arbol'].replace('_', '')
            mensaje = f"""=== Resultados del Modelo ===
                Tipo: {resultado['tipo'].capitalize()}
                {resultado['metric_name']}: {resultado['metrica']:.2%}

                === Estructura del Árbol ===
                {texto_limpio}
                """
            self.viewer.setPlainText(mensaje)

            # 3. Generar y mostrar árbol visual (DIRECTO CON TreeConverter)
            output_dir = os.path.join(self.project_dir, "output") if self.project_dir else "output"
            os.makedirs(output_dir, exist_ok=True)

            # Convertir a PDF
            pdf_path = TreeConverter.convert_to_pdf(
                texto_limpio,
                output_dir=output_dir,
                filename="decision_tree"
            )

            # Convertir PDF a imagen
            img_path = pdf_path.replace('.pdf', '.png')
            try:
                # Opción 1: Usar pdftoppm (más confiable)
                subprocess.run([
                    "pdftoppm",
                    "-png",
                    "-singlefile",
                    pdf_path,
                    os.path.join(output_dir, "temp_tree")
                ], check=True, capture_output=True)

                # Renombrar el archivo generado
                temp_img = os.path.join(output_dir, "temp_tree.png")
                if os.path.exists(temp_img):
                    os.rename(temp_img, img_path)

                # Opción 2: Si falla, usar convert (ImageMagick)
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

            # 4. Guardar resultados
            if self.project_dir:
                #guardar_modelo(resultado['modelo'], self.project_dir)
                with open(os.path.join(self.project_dir, 'arbol.txt'), 'w') as f:
                    f.write(texto_limpio)
                if img_path:
                    shutil.copy(img_path, self.project_dir)

            self.mostrar_imagen_arbol(img_path)
            QMessageBox.information(self, "Éxito", "Modelo entrenado y visualizado!")

        except Exception as e:
            error_msg = str(e)
            if "LaTeX" in error_msg:
                error_msg += "\n\nAsegúrate de tener instalados:\nsudo apt-get install texlive-latex-extra poppler-utils"
            QMessageBox.critical(self, "Error", error_msg)

    def mostrar_imagen_arbol(self, img_path):
        pixmap = QtGui.QPixmap(img_path)
        scaled_pixmap = pixmap.scaled(self.viewer2.size(), Qt.AspectRatioMode.KeepAspectRatio)
        self.viewer2.setPixmap(scaled_pixmap)

    def nuevo_proyecto(self):
        new_name, ok = QtWidgets.QInputDialog.getText(
            self,
            "Nuevo Proyecto",
            "Ingrese el nombre del nuevo proyecto:",
        )

        if ok and new_name:
            # Resetear todos los valores del proyecto actual
            self.current_project = new_name.strip()
            self.current_dataset = None
            self.current_dataset_name = ""
            self.modelo = None

            # Actualizar UI
            self.lbl_project.setText(f"Proyecto: {self.current_project}")
            self.dataset_combo.setCurrentIndex(0)
            self.viewer.clear()
            self.viewer2.clear()

            # Crear directorio
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