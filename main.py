import sys
import pandas as pd
import numpy as np

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QComboBox,
                             QListWidget, QAbstractItemView, QMessageBox, QTextEdit,
                             QFormLayout, QGroupBox, QSpinBox, QDoubleSpinBox, QTabWidget)

from dataprocess import dataprocess
from prediction import predict


class DemographyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.df = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Система демографічного прогнозування')
        self.resize(900, 680)
        self.setStyleSheet("font-size: 14px;")

        main_layout = QVBoxLayout()
        self.tabs = QTabWidget()

        self.tab_config = QWidget()
        self.tab_graph = QWidget()
        self.tab_export = QWidget()

        self.tabs.addTab(self.tab_config, "Параметри та Запуск")
        self.tabs.addTab(self.tab_graph, "Графік прогнозу")
        self.tabs.addTab(self.tab_export, "Експортувати прогноз")

        self.tabs.setTabEnabled(1, False)
        self.tabs.setTabEnabled(2, False)

        self.setup_config_tab()
        self.setup_graph_tab()
        self.setup_export_tab()

        main_layout.addWidget(self.tabs)

        main_layout.addWidget(QLabel('<b>Журнал виконання:</b>'))
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(120)
        self.log_output.setStyleSheet("background-color: #f4f4f4; font-family: Consolas;")
        main_layout.addWidget(self.log_output)

        self.setLayout(main_layout)

    def setup_config_tab(self):
        layout = QVBoxLayout()

        layout.setSpacing(10)

        self.btn_load = QPushButton('Завантажити CSV файл')
        self.btn_load.clicked.connect(self.load_data)
        self.lbl_file = QLabel('Файл не вибрано')
        layout.addWidget(self.btn_load)
        layout.addWidget(self.lbl_file)

        group_range = QGroupBox("Налаштування часових меж:")
        range_layout = QVBoxLayout()
        h_layout = QHBoxLayout()
        self.train_start = QSpinBox()
        self.train_end = QSpinBox()
        for sb in [self.train_start, self.train_end]: sb.setRange(1900, 2100); sb.setValue(2000)
        h_layout.addWidget(QLabel("Навчання з:"));
        h_layout.addWidget(self.train_start)
        h_layout.addWidget(QLabel("по:"));
        h_layout.addWidget(self.train_end)
        range_layout.addLayout(h_layout)

        f_layout = QHBoxLayout()
        self.pred_start = QSpinBox()
        self.pred_end = QSpinBox()
        for sb in [self.pred_start, self.pred_end]: sb.setRange(1900, 2100); sb.setValue(2025)
        f_layout.addWidget(QLabel("Прогноз з:"));
        f_layout.addWidget(self.pred_start)
        f_layout.addWidget(QLabel("по:"));
        f_layout.addWidget(self.pred_end)
        range_layout.addLayout(f_layout)
        group_range.setLayout(range_layout)
        layout.addWidget(group_range)

        self.spin_ks = QDoubleSpinBox()
        self.spin_ks.setRange(0.01, 2.0)
        self.spin_ks.setDecimals(3)
        self.spin_ks.setValue(0.583)

        layout.addSpacing(10)
        layout.addWidget(QLabel("<b>Коефіцієнт катаклізму (Ks):</b>"))
        layout.addWidget(self.spin_ks)

        layout.addSpacing(10)
        self.list_factors = QListWidget()
        self.list_factors.setSelectionMode(QAbstractItemView.MultiSelection)
        self.list_factors.setMaximumHeight(150)
        layout.addWidget(QLabel('<b>Оберіть економічні фактори:</b>'))
        layout.addWidget(self.list_factors)

        layout.addSpacing(15)
        self.btn_run = QPushButton('РОЗПОЧАТИ АНАЛІЗ')
        self.btn_run.setStyleSheet("font-weight: bold; padding: 10px; background-color: #e1e1e1;")
        self.btn_run.clicked.connect(self.run_analysis)
        layout.addWidget(self.btn_run)

        layout.addStretch()

        self.tab_config.setLayout(layout)

    def setup_graph_tab(self):
        layout = QVBoxLayout()

        self.figure = Figure(figsize=(10, 8), dpi=100)

        self.canvas = FigureCanvas(self.figure)

        layout.addWidget(self.canvas)
        self.tab_graph.setLayout(layout)

    def setup_export_tab(self):
        layout = QVBoxLayout()
        layout.addWidget(QLabel("<b>Експорт результатів прогнозування:</b>"))

        self.btn_csv = QPushButton("Експортувати як CSV");
        layout.addWidget(self.btn_csv)
        self.btn_csv.clicked.connect(self.export_csv)

        self.btn_xlsx = QPushButton("Експортувати як XLSX");
        layout.addWidget(self.btn_xlsx)
        self.btn_xlsx.clicked.connect(self.export_xlsx)
        
        self.btn_pdf = QPushButton("Експортувати звіт як PDF");
        layout.addWidget(self.btn_pdf)

        layout.addStretch()
        self.tab_export.setLayout(layout)

    def load_data(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Оберіть набір даних", "", "CSV Files (*.csv)",
                                                   options=options)
        if file_name:
            try:
                self.df = dataprocess(file_name)
                self.lbl_file.setText(f'Завантажено: {file_name}')
                self.list_factors.clear()
                self.list_factors.addItems(self.df.columns.tolist())
                self.log(f"Файл успішно зчитано. Знайдено {len(self.df.columns)} змінних.")
                self.tabs.setTabEnabled(1, False);
                self.tabs.setTabEnabled(2, False)
            except Exception as e:
                QMessageBox.critical(self, "Помилка", str(e))

    def run_analysis(self):
        if self.df is None:
            QMessageBox.warning(self, "Помилка", "Завантажте файл!")
            return

        try:
            t_start, t_end = self.train_start.value(), self.train_end.value()
            p_start, p_end = self.pred_start.value(), self.pred_end.value()
            years = np.arange(t_start, t_end + 1)
            years_future = np.arange(p_start, p_end + 1)
            Ks = self.spin_ks.value()
            selected_features = [item.text() for item in self.list_factors.selectedItems()]

            self.log("-" * 20)
            self.log(f"Запуск аналізу для {len(years_future)} років...")

            self.results_df = predict(self.df, Ks, years, years_future, selected_features, fig=self.figure)

            self.canvas.draw()

            self.log("Аналіз завершено. Графіки оновлено.")
            self.tabs.setTabEnabled(1, True);
            self.tabs.setTabEnabled(2, True)
            self.tabs.setCurrentIndex(1)

        except Exception as e:
            self.log(f"Помилка: {str(e)}")
            QMessageBox.critical(self, "Помилка", str(e))

    def log(self, text):
        self.log_output.append(text)
        self.log_output.verticalScrollBar().setValue(self.log_output.verticalScrollBar().maximum())

    def export_csv(self):
        if not hasattr(self, 'results_df') or self.results_df is None:
            QMessageBox.warning(self, "Помилка", "Спочатку розпочніть аналіз!")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Зберегти як CSV", "", "CSV Files (*.csv)")
        if path:
            try:
                self.results_df.to_csv(path, index=False, encoding='utf-8-sig')
                self.log(f"Збережено у CSV: {path}")
            except Exception as e:
                QMessageBox.critical(self, "Помилка", str(e))

    def export_xlsx(self):
        if not hasattr(self, 'results_df') or self.results_df is None:
            QMessageBox.warning(self, "Помилка", "Спочатку розпочніть аналіз!")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Зберегти як Excel", "", "Excel Files (*.xlsx)")
        if path:
            try:
                self.results_df.to_excel(path, index=False)
                self.log(f"Збережено у Excel: {path}")
            except Exception as e:
                QMessageBox.critical(self, "Помилка", str(e))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DemographyApp()
    ex.show()
    sys.exit(app.exec_())