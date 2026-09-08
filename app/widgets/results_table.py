"""
ResultsTable Widget for tabular display of features, spectrum estimates, and diagnostics.
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView, QLineEdit, QPushButton, QFileDialog
from PySide6.QtCore import Qt, QModelIndex
import pandas as pd


class ResultsTableWidget(QWidget):
    """
    Searchable, sortable, copyable QTableWidget wrapper with CSV Export.
    """
    def __init__(self, headers: list[str], parent=None):
        super().__init__(parent)
        self.headers = headers
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(6)
        
        # Header toolbar
        tool_layout = QHBoxLayout()
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Filter results...")
        self.search_box.textChanged.connect(self.filter_table)
        
        self.copy_btn = QPushButton("Copy")
        self.copy_btn.clicked.connect(self.copy_to_clipboard)
        
        self.export_btn = QPushButton("Export CSV")
        self.export_btn.clicked.connect(self.export_csv)
        
        tool_layout.addWidget(self.search_box)
        tool_layout.addWidget(self.copy_btn)
        tool_layout.addWidget(self.export_btn)
        
        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        
        main_layout.addLayout(tool_layout)
        main_layout.addWidget(self.table)

    def set_data(self, data: list[list[str]]):
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(data))
        for row_idx, row in enumerate(data):
            for col_idx, val in enumerate(row):
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(row_idx, col_idx, item)
        self.table.setSortingEnabled(True)

    def filter_table(self, query: str):
        query = query.lower().strip()
        for row in range(self.table.rowCount()):
            match = False
            for col in range(self.table.columnCount()):
                item = self.table.item(row, col)
                if item and query in item.text().lower():
                    match = True
                    break
            self.table.setRowHidden(row, not match)

    def copy_to_clipboard(self):
        selected_rows = []
        for r in range(self.table.rowCount()):
            if not self.table.isRowHidden(r):
                row_vals = [self.table.item(r, c).text() if self.table.item(r, c) else "" for c in range(self.table.columnCount())]
                selected_rows.append("\t".join(row_vals))
        text = "\n".join(selected_rows)
        from PySide6.QtWidgets import QApplication
        QApplication.clipboard().setText(text)

    def export_csv(self):
        filepath, _ = QFileDialog.getSaveFileName(self, "Export Results CSV", "analysis_results.csv", "CSV Files (*.csv)")
        if filepath:
            data = []
            for r in range(self.table.rowCount()):
                row_vals = [self.table.item(r, c).text() if self.table.item(r, c) else "" for c in range(self.table.columnCount())]
                data.append(row_vals)
            df = pd.DataFrame(data, columns=self.headers)
            df.to_csv(filepath, index=False)
