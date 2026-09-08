"""
ParameterPanel Widget for managing DSP & analysis input controls cleanly.
"""

from PySide6.QtWidgets import QFrame, QFormLayout, QLabel, QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit, QCheckBox
from PySide6.QtCore import Qt


class ParameterPanel(QFrame):
    """
    Card containing structured form inputs for analysis parameters.
    """
    def __init__(self, title: str = "Parameters", parent=None):
        super().__init__(parent)
        self.setObjectName("CardPanel")
        
        self.form_layout = QFormLayout(self)
        self.form_layout.setContentsMargins(14, 12, 14, 12)
        self.form_layout.setSpacing(10)
        
        if title:
            title_lbl = QLabel(title)
            title_lbl.setStyleSheet("font-size: 13px; font-weight: 700; color: #38BDF8; margin-bottom: 4px;")
            self.form_layout.addRow(title_lbl)

    def add_spinbox(self, label: str, min_val: int, max_val: int, default_val: int, step: int = 1, tooltip: str = "") -> QSpinBox:
        sb = QSpinBox()
        sb.setRange(min_val, max_val)
        sb.setValue(default_val)
        sb.setSingleStep(step)
        if tooltip:
            sb.setToolTip(tooltip)
        lbl = QLabel(label)
        if tooltip:
            lbl.setToolTip(tooltip)
        self.form_layout.addRow(lbl, sb)
        return sb

    def add_double_spinbox(self, label: str, min_val: float, max_val: float, default_val: float, step: float = 0.1, decimals: int = 2, tooltip: str = "") -> QDoubleSpinBox:
        dsb = QDoubleSpinBox()
        dsb.setRange(min_val, max_val)
        dsb.setValue(default_val)
        dsb.setSingleStep(step)
        dsb.setDecimals(decimals)
        if tooltip:
            dsb.setToolTip(tooltip)
        lbl = QLabel(label)
        if tooltip:
            lbl.setToolTip(tooltip)
        self.form_layout.addRow(lbl, dsb)
        return dsb

    def add_combobox(self, label: str, options: list[str], default_option: str = "", tooltip: str = "") -> QComboBox:
        cb = QComboBox()
        cb.addItems(options)
        if default_option and default_option in options:
            cb.setCurrentText(default_option)
        if tooltip:
            cb.setToolTip(tooltip)
        lbl = QLabel(label)
        if tooltip:
            lbl.setToolTip(tooltip)
        self.form_layout.addRow(lbl, cb)
        return cb
