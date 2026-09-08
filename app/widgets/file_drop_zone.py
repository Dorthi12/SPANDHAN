"""
FileDropZone Widget supporting drag-and-drop file imports (WAV, CSV, MAT, TXT).
"""

import os
from PySide6.QtWidgets import QFrame, QVBoxLayout, QLabel, QPushButton, QFileDialog
from PySide6.QtCore import Qt, Signal


class FileDropZone(QFrame):
    """
    Drag and drop signal file dropzone card.
    """
    file_dropped = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("CardPanelHover")
        self.setAcceptDrops(True)
        self.setStyleSheet("""
            #CardPanelHover {
                background-color: #1E293B;
                border: 2px dashed #334155;
                border-radius: 10px;
                padding: 24px;
            }
            #CardPanelHover:hover {
                border-color: #38BDF8;
                background-color: #1E3A8A20;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(10)
        
        icon_lbl = QLabel("📂")
        icon_lbl.setStyleSheet("font-size: 32px;")
        icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        title_lbl = QLabel("Drag & Drop Signal File Here")
        title_lbl.setStyleSheet("font-size: 14px; font-weight: 700; color: #F8FAFC;")
        title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        sub_lbl = QLabel("Supported formats: WAV, CSV, MAT, TXT")
        sub_lbl.setStyleSheet("font-size: 11px; color: #94A3B8;")
        sub_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.browse_btn = QPushButton("Browse File...")
        self.browse_btn.setObjectName("PrimaryButton")
        self.browse_btn.clicked.connect(self.browse_file)
        
        layout.addWidget(icon_lbl)
        layout.addWidget(title_lbl)
        layout.addWidget(sub_lbl)
        layout.addWidget(self.browse_btn, alignment=Qt.AlignmentFlag.AlignCenter)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            filepath = urls[0].toLocalFile()
            if os.path.isfile(filepath):
                self.file_dropped.emit(filepath)

    def browse_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Open Digital Signal File",
            "",
            "All Supported Files (*.wav *.csv *.mat *.txt);;WAV Files (*.wav);;CSV Files (*.csv);;MAT Files (*.mat);;TXT Files (*.txt)"
        )
        if filepath:
            self.file_dropped.emit(filepath)
