import subprocess
import sys
import os
from datetime import datetime
import json
import shutil

import numpy as np
from PyQt6 import QtWidgets, QtGui
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import QHBoxLayout, QMessageBox, QComboBox
from decision_tree_git_que_puso_el_profe import TreeConverter

from model import entrenar_arbol_decision, guardar_modelo
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

        self.btn_custom_dataset = QtWidgets.QPushButton("Cargar dataset personalizado")
        self.btn_custom_dataset.clicked.connect(self.load_custom_dataset)

        # Botones base (funcionalidad mínima)
        self.buttonOpen = QtWidgets.QPushButton("Abrir archivo")
        self.buttonOpen.setMinimumSize(QSize(200, 50))

        # Layout principal
        layout = QtWidgets.QGridLayout(self)

        # Sección de datasets
        dataset_layout = QHBoxLayout()
        dataset_layout.addWidget(self.dataset_combo)
        dataset_layout.addWidget(self.btn_custom_dataset)

        # Sección de botones base
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.buttonOpen)

        # Ensamblado final
        layout.addLayout(project_layout, 0, 0, 1, 4)
        layout.addLayout(dataset_layout, 0, 0, 1, 4)
        layout.addLayout(button_layout, 1, 0, 1, 4)
        layout.addWidget(self.viewer, 2, 0, 1, 2)
        layout.addWidget(self.viewer2, 2, 2, 1, 2)
        button_layout.addWidget(self.btn_entrenar)


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
        if self.current_project:
            # Crear directorio si no existe
            self.project_dir = os.path.join(os.getcwd(), "proyectos", self.current_project)
            os.makedirs(self.project_dir, exist_ok=True)

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
            resultado = entrenar_arbol_decision(self.current_dataset)

            texto_limpio = resultado['texto_arbol'].replace('_', '')  # Elimina guiones bajos
            resultado['texto_arbol'] = texto_limpio

            # Construir y mostrar texto del árbol
            mensaje = f"""=== Resultados del Modelo ===
                Tipo: {resultado['tipo'].capitalize()}
                {resultado['metric_name']}: {resultado['metrica']:.2%}
            
                === Estructura del Árbol ===
                {resultado['texto_arbol']}
                """
            self.viewer.setPlainText(mensaje)

            # Generar LaTeX y imagen (NUEVO)
            latex_code = self.texto_a_latex(resultado['texto_arbol'])
            output_dir = os.path.join(self.project_dir, "latex_output") if self.project_dir else "latex_output"
            img_path = self.generar_pdf_y_imagen(latex_code, output_dir, "decision_tree")

            if img_path:
                self.mostrar_imagen_arbol(img_path)

            # Guardar resultados
            if self.project_dir:
                self.guardar_resultados_completos(resultado, img_path)

            QMessageBox.information(self, "Éxito", "Modelo entrenado y visualizado!")

        except Exception as e:
            error_msg = str(e)
            QMessageBox.critical(self, "Error", error_msg)

    def mostrar_imagen_arbol(self, img_path):
        pixmap = QtGui.QPixmap(img_path)
        scaled_pixmap = pixmap.scaled(self.viewer2.size(), Qt.AspectRatioMode.KeepAspectRatio)
        self.viewer2.setPixmap(scaled_pixmap)

    def guardar_resultados_completos(self, resultado, img_path):
        # Guardar modelo
        guardar_modelo(resultado['modelo'], self.project_dir)

        # Guardar texto
        with open(os.path.join(self.project_dir, 'arbol.txt'), 'w') as f:
            f.write(resultado['texto_arbol'])

        # Guardar imagen en proyecto
        if img_path:
            img_name = os.path.basename(img_path)
            shutil.copy(img_path, os.path.join(self.project_dir, img_name))

    def texto_a_latex(self, texto_arbol):
        try:
            return TreeConverter.convert(texto_arbol, package='forest')
        except ValueError as e:
            QMessageBox.critical(self, "Error", str(e))
            return ""

    def generar_pdf_y_imagen(self, tex_code, output_dir, nombre):
        os.makedirs(output_dir, exist_ok=True)
        tex_path = os.path.join(output_dir, f"{nombre}.tex")

        try:
            with open(tex_path, 'w', encoding='utf-8') as f:
                f.write(tex_code)

            # Compilar con pdflatex (2 veces para referencias cruzadas)
            for _ in range(2):
                result = subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", f"{nombre}.tex"],
                    cwd=output_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=30
                )
                if result.returncode != 0:
                    error = result.stderr.decode('utf-8', errors='ignore')
                    raise RuntimeError(f"LaTeX error: {error[:500]}")

            # Convertir a imagen
            pdf_path = os.path.join(output_dir, f"{nombre}.pdf")
            img_path = os.path.join(output_dir, f"{nombre}.png")

            subprocess.run(
                ["pdftoppm", "-png", "-singlefile", pdf_path, img_path.replace('.png', '')],
                check=True
            )

            return img_path if os.path.exists(img_path) else None

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error generating PDF: {str(e)}")
            return None

    def handleOpen(self):
        pass

    def ProcesarImage(self):
        pass

    def handleSaveFile(self):
        pass

    def ActualizarPixMap2(self, image):
        pass



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec())