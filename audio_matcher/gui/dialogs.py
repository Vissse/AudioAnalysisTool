import io
import sys
import ctypes
import numpy as np
from ctypes.wintypes import HWND, DWORD

from PyQt6.QtWidgets import (QDialog, QLabel, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QTabWidget, QWidget, QComboBox, 
                             QFormLayout, QMessageBox, QSlider, QGroupBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtMultimedia import QMediaDevices  # PŘIDÁNO PRO ZVUKOVÁ ZAŘÍZENÍ

from matplotlib.figure import Figure
import librosa.display

from config import AUDIO_CFG, APP_CFG, MODEL_CFG, UI_CFG, save_settings


class SettingsDialog(QDialog):
    """Modální okno pro konfiguraci AI modelů a prahových hodnot aplikace."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_app = parent  # Uložení reference na hlavní aplikaci pro změnu zvuku
        self.setWindowTitle("Nastavení Aplikace")
        self.resize(500, 520) # Mírně zvětšeno, aby se vešly nové GroupBoxy
        
        # Aplikace stylování pro moderní Slidery specificky v tomto okně
        self.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #d1d5db;
                height: 6px;
                background: #f3f4f6;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #10b981;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #ffffff;
                border: 1px solid #9ca3af;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #f9fafb;
                border: 2px solid #10b981;
            }
        """)
        
        self.init_ui()
        self.apply_native_titlebar_color()

    def apply_native_titlebar_color(self, hex_bg="#10b981", hex_text="#ffffff"):
        """DWM API pro zelenou lištu okna Windows 11."""
        if sys.platform != "win32":
            return
        try:
            DWMWA_CAPTION_COLOR = 35
            DWMWA_TEXT_COLOR = 36
            hwnd = HWND(int(self.winId()))
            
            bg = hex_bg.lstrip('#')
            bg_colorref = (int(bg[4:6], 16) << 16) | (int(bg[2:4], 16) << 8) | int(bg[0:2], 16)
            
            txt = hex_text.lstrip('#')
            txt_colorref = (int(txt[4:6], 16) << 16) | (int(txt[2:4], 16) << 8) | int(txt[0:2], 16)
            
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, DWORD(DWMWA_CAPTION_COLOR), ctypes.byref(DWORD(bg_colorref)), ctypes.sizeof(DWORD))
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, DWORD(DWMWA_TEXT_COLOR), ctypes.byref(DWORD(txt_colorref)), ctypes.sizeof(DWORD))
        except Exception as e:
            print(f"DWM Titlebar color not applied on Dialog: {e}")

    def _create_slider_row(self, initial_val: float, max_val: float = 1.0) -> tuple[QSlider, QLabel, QHBoxLayout]:
        """Tvoří mapovaný slider pro desetinná čísla (Qt umí jen int)."""
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, int(max_val * 100))
        slider.setValue(int(initial_val * 100))
        slider.setCursor(Qt.CursorShape.PointingHandCursor)
        
        lbl_val = QLabel(f"{initial_val:.2f}")
        lbl_val.setFixedWidth(35)
        lbl_val.setStyleSheet("font-weight: bold; color: #10b981;")
        
        # Lambda zachytí změnu integeru a přepíše label na float
        slider.valueChanged.connect(lambda v: lbl_val.setText(f"{v/100:.2f}"))
        
        layout = QHBoxLayout()
        layout.addWidget(slider)
        layout.addWidget(lbl_val)
        return slider, lbl_val, layout

    def init_ui(self):
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        
        # --- TAB 1: Základní konfigurace ---
        tab_engine = QWidget()
        form_engine = QFormLayout(tab_engine)
        form_engine.setSpacing(15)
        form_engine.setContentsMargins(15, 20, 15, 15)
        
        # Výběr zvukového výstupu
        self.combo_audio_device = QComboBox()
        self.combo_audio_device.setCursor(Qt.CursorShape.PointingHandCursor)
        
        current_device_name = self.parent_app.audio_output.device().description() if self.parent_app else ""
        index_to_select = 0
        
        for i, device in enumerate(QMediaDevices.audioOutputs()):
            self.combo_audio_device.addItem(device.description(), device)
            if device.description() == current_device_name:
                index_to_select = i
                
        self.combo_audio_device.setCurrentIndex(index_to_select)
        self.combo_audio_device.currentIndexChanged.connect(self.change_audio_device)
        
        # Výběr jazyka Whisperu (ODEBRÁNO AUTOMATICKY)
        self.combo_language = QComboBox()
        self.combo_language.addItem("Čeština", "cs")
        self.combo_language.addItem("Angličtina", "en")
        self.combo_language.addItem("Slovenština", "sk")
        self.combo_language.addItem("Němčina", "de")
        self.combo_language.setCursor(Qt.CursorShape.PointingHandCursor)
        
        index_lang = self.combo_language.findData(MODEL_CFG.whisper_language)
        if index_lang >= 0:
            self.combo_language.setCurrentIndex(index_lang)
        elif MODEL_CFG.whisper_language == "":
            # Pokud v configu zůstalo prázdné "", nastaví se výchozí Čeština
            self.combo_language.setCurrentIndex(0)

        self.cb_device = QComboBox()
        self.cb_device.addItems(["cpu", "cuda"])
        self.cb_device.setCurrentText(MODEL_CFG.device)
        self.cb_device.setCursor(Qt.CursorShape.PointingHandCursor)
        
        form_engine.addRow("Zvukový výstup:", self.combo_audio_device)
        form_engine.addRow("Jazyk (Whisper):", self.combo_language)
        form_engine.addRow("Výpočetní jednotka (GPU/cpu):", self.cb_device)
        
        # --- TAB 2: Filtry a Algoritmy ---
        tab_algo = QWidget()
        layout_algo = QVBoxLayout(tab_algo)
        layout_algo.setSpacing(15)
        layout_algo.setContentsMargins(15, 15, 15, 15)
        
        # ODRÁŽKA 1: Filtry a předzpracování (Přesunuto z první záložky)
        grp_filters = QGroupBox("Filtry a předzpracování signálu")
        grp_filters.setStyleSheet("QGroupBox { font-weight: bold; color: #4b5563; }")
        form_filters = QFormLayout(grp_filters)
        form_filters.setSpacing(15)
        
        self.cb_vad = QComboBox()
        self.cb_vad.addItems(["Silero VAD", "RMS"])
        self.cb_vad.setCurrentText(MODEL_CFG.vad_method)
        self.cb_vad.setCursor(Qt.CursorShape.PointingHandCursor)
        
        self.slider_silero, _, lay_silero = self._create_slider_row(MODEL_CFG.silero_vad_threshold)
        self.slider_whisper, _, lay_whisper = self._create_slider_row(APP_CFG.whisper_min_confidence)
        
        form_filters.addRow("Metoda detekce hlasu (VAD):", self.cb_vad)
        form_filters.addRow("Práh důvěry Silero VAD:", lay_silero)
        form_filters.addRow("Minimální jistota Whisperu:", lay_whisper)
        layout_algo.addWidget(grp_filters)
        
        # ODRÁŽKA 2: Parametry srovnávacích algoritmů
        grp_dtw = QGroupBox("Parametry vyhledávacích algoritmů")
        grp_dtw.setStyleSheet("QGroupBox { font-weight: bold; color: #4b5563; }")
        form_dtw = QFormLayout(grp_dtw)
        form_dtw.setSpacing(15)

        self.slider_dtw_stop, _, lay_dtw = self._create_slider_row(APP_CFG.dtw_early_stop_threshold, max_val=2.0)
        
        form_dtw.addRow("DTW zahazovací práh:", lay_dtw)
        layout_algo.addWidget(grp_dtw)
        
        layout_algo.addStretch()

        # Přidání záložek do okna
        self.tabs.addTab(tab_engine, "Obecná nastavení")
        self.tabs.addTab(tab_algo, "Filtry a algoritmy")
        
        # --- Tlačítka ---
        btn_layout = QHBoxLayout()
        btn_save = QPushButton("Uložit nastavení")
        btn_save.setObjectName("primaryButton")
        btn_save.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_save.clicked.connect(self.save_and_close)
        
        btn_cancel = QPushButton("Zrušit")
        btn_cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_cancel.clicked.connect(self.reject)
        
        btn_layout.addStretch()
        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(btn_save)
        
        layout.addWidget(self.tabs)
        layout.addLayout(btn_layout)

    def change_audio_device(self, index):
        """Změní výstupní zařízení přímo v hlavní aplikaci."""
        if index >= 0 and self.parent_app:
            device = self.combo_audio_device.itemData(index)
            self.parent_app.audio_output.setDevice(device)
            if hasattr(self.parent_app, 'log_status'):
                self.parent_app.log_status(f"Zvukový výstup změněn na: {device.description()}")

    def save_and_close(self):
        # Aplikace hodnot do RAM přes slider mapování
        MODEL_CFG.whisper_language = self.combo_language.currentData()
        MODEL_CFG.device = self.cb_device.currentText()
        MODEL_CFG.vad_method = self.cb_vad.currentText()
        MODEL_CFG.silero_vad_threshold = self.slider_silero.value() / 100.0
        
        APP_CFG.whisper_min_confidence = self.slider_whisper.value() / 100.0
        APP_CFG.dtw_early_stop_threshold = self.slider_dtw_stop.value() / 100.0
        
        save_settings()
        QMessageBox.information(self, "Uloženo", "Nastavení bylo úspěšně uloženo.\nPři změně vypočetní jednotky (GPU/cpu) je nutný restart aplikace.")
        self.accept()


class VisualCompareDialog(QDialog):
    def __init__(self, res, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Detailní analýza")
        self.resize(650, 380)
        
        # 1. Změna barvy horní lišty na zelenou (Windows 11)
        self.apply_native_titlebar_color()

        # 2. Světlý vzhled dialogu (bílo-šedé pozadí)
        self.setStyleSheet("""
            QDialog { background-color: #f9fafb; }
            QLabel { color: #1f2937; }
            QPushButton { 
                background-color: #ffffff; 
                border: 1px solid #d1d5db; 
                border-radius: 4px; 
                padding: 6px 15px; 
                color: #4b5563; 
                font-weight: 500; 
                font-size: 10pt; 
            }
            QPushButton:hover { 
                background-color: #e5e7eb; 
                border: 1px solid #9ca3af; 
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        
        title = QLabel(f"Metoda: {res.method.upper()} | Skóre vzdálenosti: {res.score:.4f}")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #10b981; margin-bottom: 5px;")
        layout.addWidget(title, alignment=Qt.AlignmentFlag.AlignCenter)

        images_container = QHBoxLayout()
        images_container.setSpacing(20)

        frames_per_sec = res.sr / AUDIO_CFG.hop_length

        def add_black_padding(spec, padding_sec=0.15):
            pad_frames = int(padding_sec * frames_per_sec)
            min_db = np.min(spec)
            return np.pad(spec, ((0, 0), (pad_frames, pad_frames)), mode='constant', constant_values=min_db)

        spec_q = res.vis_spec_query
        spec_q_padded = add_black_padding(spec_q)
        dur_q = spec_q_padded.shape[1] / frames_per_sec
        
        start_idx = int(res.start_f)
        end_idx = int(res.end_f)
        spec_m = res.vis_spec_long[:, start_idx:end_idx]
        spec_m_padded = add_black_padding(spec_m)
        dur_m = spec_m_padded.shape[1] / frames_per_sec

        pix_q = self.spec_to_pixmap(spec_q_padded, res.sr, dur_q)
        pix_m = self.spec_to_pixmap(spec_m_padded, res.sr, dur_m)

        def create_spec_widget(pixmap, title_text):
            w = QWidget()
            l = QVBoxLayout(w)
            l.setContentsMargins(0, 0, 0, 0)
            
            lbl_t = QLabel(title_text)
            # Tmavý text popisků, aby byl na bílém pozadí vidět
            lbl_t.setStyleSheet("color: #4b5563; font-size: 11px; font-weight: bold;")
            
            lbl_i = QLabel()
            lbl_i.setPixmap(pixmap)
            
            l.addWidget(lbl_t, alignment=Qt.AlignmentFlag.AlignCenter)
            l.addWidget(lbl_i, alignment=Qt.AlignmentFlag.AlignCenter) 
            return w

        images_container.addStretch(1)
        images_container.addWidget(create_spec_widget(pix_q, "Hledaný vzorek"))
        images_container.addWidget(create_spec_widget(pix_m, f"Nalezený úsek (Čas: {res.start_f / frames_per_sec:.2f} s)"))
        images_container.addStretch(1)
        
        layout.addLayout(images_container)
        
        btn_close = QPushButton("Zavřít")
        btn_close.setFixedSize(120, 35)
        btn_close.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close, alignment=Qt.AlignmentFlag.AlignCenter)

    # 3. Přidání funkce pro obarvení nativní Windows lišty do tohoto dialogu
    def apply_native_titlebar_color(self, hex_bg="#10b981", hex_text="#ffffff"):
        if sys.platform != "win32":
            return
        try:
            DWMWA_CAPTION_COLOR = 35
            DWMWA_TEXT_COLOR = 36
            hwnd = HWND(int(self.winId()))
            
            bg = hex_bg.lstrip('#')
            bg_colorref = (int(bg[4:6], 16) << 16) | (int(bg[2:4], 16) << 8) | int(bg[0:2], 16)
            
            txt = hex_text.lstrip('#')
            txt_colorref = (int(txt[4:6], 16) << 16) | (int(txt[2:4], 16) << 8) | int(txt[0:2], 16)
            
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, DWORD(DWMWA_CAPTION_COLOR), ctypes.byref(DWORD(bg_colorref)), ctypes.sizeof(DWORD))
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, DWORD(DWMWA_TEXT_COLOR), ctypes.byref(DWORD(txt_colorref)), ctypes.sizeof(DWORD))
        except Exception as e:
            print(f"Nepodařilo se obarvit horní lištu: {e}")

    # 4. Změna barvy pozadí Matplotlib plátna, aby splynulo s oknem
    def spec_to_pixmap(self, spec: np.ndarray, sr: int, duration: float) -> QPixmap:
        width_inches = 0.6 + ((duration / 10.0) * 5.0) 
        
        # Facecolor odpovídá background-color v CSS (#f9fafb)
        fig = Figure(figsize=(width_inches, 3.5), dpi=100, facecolor='#f9fafb')
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        
        librosa.display.specshow(spec, ax=ax, cmap='magma', sr=sr, hop_length=AUDIO_CFG.hop_length)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), pad_inches=0)
        buf.seek(0)
        return QPixmap.fromImage(QImage.fromData(buf.getvalue()))