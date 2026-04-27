# main.py
import sys
import os
import ctypes
import time
from PyQt6.QtWidgets import QApplication, QMessageBox, QWidget, QVBoxLayout, QLabel, QProgressBar, QFrame
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QIcon, QPainter, QColor

# 1. Trik pro Windows Taskbar (aby měla aplikace správnou ikonu i na dolní liště)
if sys.platform == "win32":
    try:
        myappid = 'mycompany.audiomatcher.tool.1.0'
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception:
        pass

# Vytvoření aplikace (musí být úplně první před jakýmkoliv GUI)
app = QApplication(sys.argv)

# Import konfigurace a cest k ikonám (try-except pro jistotu, kdyby config zlobil)
try:
    from config import APP_CFG
    # Nastavení ikony okna/taskbaru (zůstává původní .ico)
    if os.path.exists(APP_CFG.icon_path):
        app.setWindowIcon(QIcon(APP_CFG.icon_path))
except Exception:
    pass


class ModernSplashScreen(QWidget):
    """Moderní načítací okno se zakulacenými rohy a ukazatelem průběhu."""
    def __init__(self):
        super().__init__()
        # Skryjeme výchozí Windows rámeček a nastavíme průhledné pozadí
        self.setWindowFlags(Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(450, 320)
        
        # Hlavní kontejner (zde děláme zakulacené rohy a bílé pozadí)
        self.bg_frame = QFrame(self)
        self.bg_frame.setObjectName("bgFrame")
        self.bg_frame.setFixedSize(450, 320)
        
        layout = QVBoxLayout(self.bg_frame)
        layout.setContentsMargins(40, 30, 40, 40) # Mírně upraven horní margin
        layout.setSpacing(10) # Menší spacing pro kompaktnější vzhled horní části
        
        # --- NAČTENÍ A PŘEBARVENÍ IKONY NA ČERNO ---
        self.lbl_logo = QLabel()
        self.lbl_logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        try:
            # Načte tvou stávající ikonu (nebo ikona.png, pokud ji máš v configu)
            pixmap = QPixmap(APP_CFG.icon_path) 
            
            if not pixmap.isNull():
                # Vytvoříme "malíře", který bude kreslit na náš obrázek
                painter = QPainter(pixmap)
                # Nastavíme mód "SourceIn" = kresli jen tam, kde už něco je (vynechá průhledné pozadí)
                painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
                # Vybarví celý obdélník černou barvou
                painter.fillRect(pixmap.rect(), QColor("black"))
                painter.end() # Ukončíme kreslení
                
                # Zvětšíme/zmenšíme přebarvené logo na správnou velikost
                self.lbl_logo.setPixmap(pixmap.scaledToHeight(80, Qt.TransformationMode.SmoothTransformation))
        except Exception as e:
            print(f"Chyba při přebarvování loga: {e}")
            pass
            
        # Nadpis aplikace (zůstává stejný)
        self.lbl_title = QLabel("Audio Analysis Tool")
        self.lbl_title.setObjectName("title")
        self.lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Sestavení horní části (Logo -> Nadpis)
        layout.addStretch(1)
        layout.addWidget(self.lbl_logo)
        layout.addWidget(self.lbl_title)
        layout.addStretch(1)
        
        # Textový popisek, co se zrovna děje
        self.lbl_status = QLabel("Spouštím aplikaci...")
        self.lbl_status.setObjectName("status")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Progress bar v zeleném stylu
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        
        layout.addWidget(self.lbl_status)
        layout.addWidget(self.progress)
        
        # Aplikace CSS stylů pro tento konkrétní SplashScreen
        self.setStyleSheet("""
            QFrame#bgFrame {
                background-color: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 12px;
            }
            QLabel#title {
                color: #1f2937;
                font-size: 18pt;
                font-weight: bold;
                margin-top: 5px; /* Zmenšen margin nad nadpisem */
            }
            QLabel#status {
                color: #6b7280;
                font-size: 10pt;
                font-weight: 500;
                margin-bottom: 5px;
            }
            QProgressBar {
                border: none;
                background-color: #e5e7eb;
                border-radius: 3px;
                height: 6px;
            }
            QProgressBar::chunk {
                background-color: #10b981;
                border-radius: 3px;
            }
        """)

    def update_progress(self, value, text):
        """Aktualizuje lištu a donutí GUI k okamžitému překreslení."""
        self.progress.setValue(value)
        self.lbl_status.setText(text)
        QApplication.processEvents() 
        time.sleep(0.05)


if __name__ == '__main__':
    # Zobrazíme načítací okno hned jako první věc
    splash = ModernSplashScreen()
    splash.show()
    
    try:
        # Plynule posouváme progress bar a na pozadí taháme těžké importy
        splash.update_progress(10, "Načítám konfigurační profily...")
        
        splash.update_progress(30, "Zavádím matematické knihovny (Numpy)...")
        import numpy as np
        
        splash.update_progress(50, "Iniciuji moduly pro zpracování zvuku...")
        import soundfile as sf
        import librosa
        
        splash.update_progress(70, "Příprava AI subsystémů (PyTorch, Whisper)...")
        # Natažení gui.app trvá nejdéle, protože importuje všechny core moduly
        from gui.app import AudioMatcherApp
        
        splash.update_progress(90, "Sestavuji uživatelské prostředí...")
        window = AudioMatcherApp()
        
        splash.update_progress(100, "Dokončeno. Otevírám aplikaci...")
        time.sleep(0.4) # Drobná pauza, aby uživatel postřehl 100 %
        
    except Exception as e:
        # Pokud něco selže, zavřeme splash a ukážeme klasický error
        splash.close()
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setWindowTitle("Chyba spuštění")
        msg.setText("Došlo k chybě při inicializaci jádra aplikace.")
        msg.setInformativeText(f"Detail:\n{e}")
        msg.exec()
        sys.exit(1)

    # Zavřeme načítací okno a ukážeme to hlavní
    splash.close()
    window.show()
    sys.exit(app.exec())