# ==============================================================================
# audio_matcher/main.py
# ==============================================================================
import sys
import os
import ctypes
import time
from PyQt6.QtWidgets import QApplication, QMessageBox, QWidget, QVBoxLayout, QLabel, QProgressBar, QFrame
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QIcon, QPainter, QColor

if sys.platform == "win32":
    try:
        myappid = 'mycompany.audiomatcher.tool.1.0'
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception:
        pass

app = QApplication(sys.argv)

try:
    from config import APP_CFG
    if os.path.exists(APP_CFG.icon_path):
        app.setWindowIcon(QIcon(APP_CFG.icon_path))
except Exception:
    pass

class ModernSplashScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(450, 320)
        
        self.bg_frame = QFrame(self)
        self.bg_frame.setObjectName("bgFrame")
        self.bg_frame.setFixedSize(450, 320)
        
        layout = QVBoxLayout(self.bg_frame)
        layout.setContentsMargins(40, 30, 40, 40)
        layout.setSpacing(10) 
        
        self.lbl_logo = QLabel()
        self.lbl_logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        try:
            pixmap = QPixmap(APP_CFG.icon_path) 
            if not pixmap.isNull():
                colored_pixmap = QPixmap(pixmap.size())
                colored_pixmap.fill(Qt.GlobalColor.transparent)
                painter = QPainter(colored_pixmap)
                painter.fillRect(colored_pixmap.rect(), QColor("#1f2937")) 
                painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationIn)
                painter.drawPixmap(0, 0, pixmap)
                painter.end()
                self.lbl_logo.setPixmap(colored_pixmap.scaledToHeight(70, Qt.TransformationMode.SmoothTransformation))
        except Exception:
            pass
            
        self.lbl_title = QLabel("Audio Analysis Tool")
        self.lbl_title.setObjectName("title")
        self.lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        layout.addStretch(1)
        layout.addWidget(self.lbl_logo)
        layout.addWidget(self.lbl_title)
        layout.addStretch(1)
        
        self.lbl_status = QLabel("Spouštím aplikaci...")
        self.lbl_status.setObjectName("status")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
    
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        
        layout.addWidget(self.lbl_status)
        layout.addWidget(self.progress)
        
        self.setStyleSheet("""
            QFrame#bgFrame { background-color: #ffffff; border: 1px solid #e5e7eb; border-radius: 12px; }
            QLabel#title { color: #1f2937; font-size: 18pt; font-weight: bold; margin-top: 5px; }
            QLabel#status { color: #6b7280; font-size: 10pt; font-weight: 500; margin-bottom: 5px; }
            QProgressBar { border: none; background-color: #e5e7eb; border-radius: 3px; height: 6px; }
            QProgressBar::chunk { background-color: #10b981; border-radius: 3px; }
        """)

    def update_progress(self, value, text):
        self.progress.setValue(value)
        self.lbl_status.setText(text)
        QApplication.processEvents() 
        time.sleep(0.05)


if __name__ == '__main__':
    splash = ModernSplashScreen()
    splash.show()
    
    try:
        splash.update_progress(10, "Načítám konfigurační profily...")
        
        splash.update_progress(30, "Zavádím matematické knihovny (Numpy)...")
        import numpy as np
        
        splash.update_progress(50, "Iniciuji moduly pro zpracování zvuku...")
        import soundfile as sf
        import librosa
        
        splash.update_progress(70, "Příprava subsystémů (PyTorch, Whisper)...")
        from gui.app import AudioMatcherApp
        
        splash.update_progress(90, "Sestavuji uživatelské prostředí...")
        window = AudioMatcherApp()
        
        splash.update_progress(100, "Dokončeno. Otevírám aplikaci...")
        time.sleep(0.4) 
        
    except Exception as e:
        splash.close()
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setWindowTitle("Chyba spuštění")
        msg.setText("Došlo k chybě při inicializaci jádra aplikace.")
        msg.setInformativeText(f"Detail:\n{e}")
        msg.exec()
        sys.exit(1)

    splash.close()
    window.show()
    sys.exit(app.exec())