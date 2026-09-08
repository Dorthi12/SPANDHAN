"""
Spandhan Application Design System & QSS Stylesheet.
Provides dark-neutral scientific engineering workstation visual design.
"""

DARK_THEME_QSS = """
/* Global Application Style */
QWidget {
    background-color: #0F172A;
    color: #F8FAFC;
    font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    font-size: 13px;
    selection-background-color: #2563EB;
    selection-color: #FFFFFF;
}

/* Main Window & Central Widget */
QMainWindow, QStackedWidget {
    background-color: #0F172A;
}

/* ScrollBars */
QScrollBar:vertical {
    border: none;
    background: #1E293B;
    width: 8px;
    margin: 0px;
    border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: #475569;
    min-height: 20px;
    border-radius: 4px;
}
QScrollBar::handle:vertical:hover {
    background: #64748B;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    border: none;
    background: #1E293B;
    height: 8px;
    margin: 0px;
    border-radius: 4px;
}
QScrollBar::handle:horizontal {
    background: #475569;
    min-width: 20px;
    border-radius: 4px;
}
QScrollBar::handle:horizontal:hover {
    background: #64748B;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

/* Sidebar Styling */
#SidebarWidget {
    background-color: #0B0F19;
    border-right: 1px solid #1E293B;
    min-width: 230px;
    max-width: 230px;
}

#SidebarLogoLabel {
    font-size: 18px;
    font-weight: 700;
    color: #38BDF8;
    letter-spacing: 1.5px;
    padding: 10px 0px 2px 0px;
}

#SidebarSubtitleLabel {
    font-size: 10px;
    color: #94A3B8;
    letter-spacing: 0.5px;
}

#NavButton {
    background-color: transparent;
    color: #94A3B8;
    text-align: left;
    padding: 10px 14px;
    border: none;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 500;
}
#NavButton:hover {
    background-color: #1E293B;
    color: #F8FAFC;
}
#NavButton[active="true"] {
    background-color: #1E3A8A;
    color: #38BDF8;
    font-weight: 600;
    border-left: 3px solid #38BDF8;
}

/* Top Bar Styling */
#TopBarWidget {
    background-color: #0B0F19;
    border-bottom: 1px solid #1E293B;
    padding: 8px 16px;
    min-height: 54px;
    max-height: 54px;
}

#PageTitleLabel {
    font-size: 16px;
    font-weight: 700;
    color: #F8FAFC;
}

#PageDescLabel {
    font-size: 11px;
    color: #94A3B8;
}

#SignalNameBadge {
    background-color: #1E293B;
    color: #E2E8F0;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 12px;
    font-weight: 500;
}

/* Cards & Panels */
#CardPanel {
    background-color: #1E293B;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 14px;
}

#CardPanelHover:hover {
    border-color: #475569;
}

#MetricCard {
    background-color: #1E293B;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 12px 14px;
}

#MetricTitle {
    font-size: 11px;
    font-weight: 600;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

#MetricValue {
    font-size: 20px;
    font-weight: 700;
    color: #38BDF8;
    margin-top: 2px;
}

#MetricSub {
    font-size: 11px;
    color: #64748B;
}

/* Buttons */
QPushButton {
    background-color: #1E293B;
    color: #F8FAFC;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #334155;
    border-color: #475569;
}
QPushButton:pressed {
    background-color: #0F172A;
}
QPushButton:disabled {
    background-color: #0F172A;
    color: #475569;
    border-color: #1E293B;
}

QPushButton#PrimaryButton {
    background-color: #2563EB;
    color: #FFFFFF;
    border: none;
    font-weight: 600;
}
QPushButton#PrimaryButton:hover {
    background-color: #1D4ED8;
}
QPushButton#PrimaryButton:pressed {
    background-color: #1E40AF;
}

QPushButton#SuccessButton {
    background-color: #059669;
    color: #FFFFFF;
    border: none;
    font-weight: 600;
}
QPushButton#SuccessButton:hover {
    background-color: #047857;
}

QPushButton#AccentButton {
    background-color: #0D9488;
    color: #FFFFFF;
    border: none;
    font-weight: 600;
}
QPushButton#AccentButton:hover {
    background-color: #0F766E;
}

/* Input Fields & Combo Boxes */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #0F172A;
    color: #F8FAFC;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 12px;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border-color: #2563EB;
}
QComboBox::drop-down {
    border: none;
    width: 20px;
}
QComboBox QAbstractItemView {
    background-color: #1E293B;
    border: 1px solid #334155;
    selection-background-color: #2563EB;
}

/* Tab Widgets */
QTabWidget::pane {
    border: 1px solid #334155;
    border-radius: 6px;
    background-color: #1E293B;
    top: -1px;
}
QTabBar::tab {
    background-color: #0F172A;
    color: #94A3B8;
    border: 1px solid #334155;
    border-bottom: none;
    padding: 8px 16px;
    margin-right: 2px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    font-weight: 500;
}
QTabBar::tab:selected {
    background-color: #1E293B;
    color: #38BDF8;
    font-weight: 600;
    border-top: 2px solid #38BDF8;
}
QTabBar::tab:hover:!selected {
    background-color: #1E293B;
    color: #E2E8F0;
}

/* Tables */
QTableWidget {
    background-color: #0F172A;
    border: 1px solid #334155;
    gridline-color: #1E293B;
    border-radius: 6px;
}
QTableWidget::item {
    padding: 6px 10px;
    border-bottom: 1px solid #1E293B;
}
QTableWidget::item:selected {
    background-color: #1E3A8A;
    color: #FFFFFF;
}
QHeaderView::section {
    background-color: #1E293B;
    color: #94A3B8;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    border: none;
    border-bottom: 1px solid #334155;
    padding: 8px;
}

/* GroupBox */
QGroupBox {
    border: 1px solid #334155;
    border-radius: 6px;
    margin-top: 12px;
    font-weight: 600;
    color: #E2E8F0;
    padding-top: 10px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
    color: #38BDF8;
}

/* Progress Bar */
QProgressBar {
    border: 1px solid #334155;
    border-radius: 4px;
    text-align: center;
    background-color: #0F172A;
    color: #F8FAFC;
}
QProgressBar::chunk {
    background-color: #2563EB;
    border-radius: 3px;
}
"""
