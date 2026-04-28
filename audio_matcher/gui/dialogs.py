# ==============================================================================
# audio_matcher/gui/dialogs.py
# ==============================================================================
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
from PyQt6.QtMultimedia import QMediaDevices 
from matplotlib.figure import Figure
import librosa.display

from config import AUDIO_CFG, APP_CFG, MODEL_CFG, save_settings

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_app = parent
        self.setWindowTitle("Nastavení Aplikace")
        self.resize(500, 520)
        
        self.setStyleSheet("""
            QSlider::groove:horizontal { border: 1px solid #d1d5db; height: 6px; background: #f3f4f6; border-radius: 3px; }
            QSlider::sub-page:horizontal { background: #10b981; border-radius: 3px; }
            QSlider::handle:horizontal { background: #ffffff; border: 1px solid #9ca3af; width: 16px; height: 16px; margin: -5px 0; border-radius: 8px; }
            QSlider::handle:horizontal:hover { background: #f9fafb; border: 2px solid #10b981; }
        """)
        
        self.init_ui()
        self.apply_native_titlebar_color()

    def apply_native_titlebar_color(self, hex_bg="#10b981", hex_text="#ffffff"):
        if sys.platform != "win32": return
        try:
            DWMWA_CAPTION_COLOR, DWMWA_TEXT_COLOR = 35, 36
            hwnd = HWND(int(self.winId()))
            bg = hex_bg.lstrip('#')
            bg_colorref = (int(bg[4:6], 16) << 16) | (int(bg[2:4], 16) << 8) | int(bg[0:2], 16)
            txt = hex_text.lstrip('#')
            txt_colorref = (int(txt[4:6], 16) << 16) | (int(txt[2:4], 16) << 8) | int(txt[0:2], 16)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, DWORD(DWMWA_CAPTION_COLOR), ctypes.byref(DWORD(bg_colorref)), ctypes.sizeof(DWORD))
            ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, DWORD(DWMWA_TEXT_COLOR), ctypes.byref(DWORD(txt_colorref)), ctypes.sizeof(DWORD))
        except Exception:
            pass

    def _create_slider_row(self, initial_val: float, max_val: float = 1.0) -> tuple[QSlider, QLabel, QHBoxLayout]:
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, int(max_val * 100))
        slider.setValue(int(initial_val * 100))
        slider.setCursor(Qt.CursorShape.PointingHandCursor)
        lbl_val = QLabel(f"{initial_val:.2f}")
        lbl_val.setFixedWidth(35)
        lbl_val.setStyleSheet("font-weight: bold; color: #10b981;")
        slider.valueChanged.connect(lambda v: lbl_val.setText(f"{v/100:.2f}"))
        layout = QHBoxLayout()
        layout.addWidget(slider)
        layout.addWidget(lbl_val)
        return slider, lbl_val, layout

    def init_ui(self):
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        
        tab_engine = QWidget()
        form_engine = QFormLayout(tab_engine)
        form_engine.setSpacing(15)
        form_engine.setContentsMargins(15, 20, 15, 15)
        
        self.combo_audio_device = QComboBox()
        current_device_name = self.parent_app.audio_output.device().description() if self.parent_app else ""
        index_to_select = 0
        for i, device in enumerate(QMediaDevices.audioOutputs()):
            self.combo_audio_device.addItem(device.description(), device)
            if device.description() == current_device_name: index_to_select = i
        self.combo_audio_device.setCurrentIndex(index_to_select)
        self.combo_audio_device.currentIndexChanged.connect(self.change_audio_device)
        
        self.combo_language = QComboBox()
        self.combo_language.addItems(["Čeština", "Angličtina", "Slovenština", "Němčina"])
        self.combo_language.setItemData(0, "cs"); self.combo_language.setItemData(1, "en")
        self.combo_language.setItemData(2, "sk"); self.combo_language.setItemData(3, "de")
        index_lang = self.combo_language.findData(MODEL_CFG.whisper_language)
        self.combo_language.setCurrentIndex(index_lang if index_lang >= 0 else 0)

        self.cb_device = QComboBox()
        self.cb_device.addItems(["cpu", "cuda"])
        self.cb_device.setCurrentText(MODEL_CFG.device)
        
        form_engine.addRow("Zvukový výstup:", self.combo_audio_device)
        form_engine.addRow("Jazyk (Whisper):", self.combo_language)
        form_engine.addRow("Výpočetní jednotka (GPU/cpu):", self.cb_device)
        
        tab_algo = QWidget()
        layout_algo = QVBoxLayout(tab_algo)
        layout_algo.setContentsMargins(15, 15, 15, 15)

        grp_filters = QGroupBox("Filtry a předzpracování signálu")
        form_filters = QFormLayout(grp_filters)
        self.slider_silero, _, lay_silero = self._create_slider_row(MODEL_CFG.silero_vad_threshold)
        self.slider_whisper, _, lay_whisper = self._create_slider_row(APP_CFG.whisper_min_confidence)
        form_filters.addRow("Práh důvěry Silero VAD:", lay_silero)
        form_filters.addRow("Minimální jistota Whisperu:", lay_whisper)
        layout_algo.addWidget(grp_filters)
        
        grp_dtw = QGroupBox("Parametry vyhledávacích algoritmů")
        form_dtw = QFormLayout(grp_dtw)
        self.slider_dtw_stop, _, lay_dtw = self._create_slider_row(APP_CFG.dtw_early_stop_threshold, max_val=2.0)
        form_dtw.addRow("DTW zahazovací práh:", lay_dtw)
        layout_algo.addWidget(grp_dtw)
        layout_algo.addStretch()

        self.tabs.addTab(tab_engine, "Obecná nastavení")
        self.tabs.addTab(tab_algo, "Filtry a algoritmy")
        
        btn_layout = QHBoxLayout()
        btn_save = QPushButton("Uložit nastavení")
        btn_save.setObjectName("primaryButton")
        btn_save.clicked.connect(self.save_and_close)
        btn_cancel = QPushButton("Zrušit")
        btn_cancel.clicked.connect(self.reject)
        
        btn_layout.addStretch()
        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(btn_save)
        
        layout.addWidget(self.tabs)
        layout.addLayout(btn_layout)

    def change_audio_device(self, index):
        if index >= 0 and self.parent_app:
            device = self.combo_audio_device.itemData(index)
            self.parent_app.audio_output.setDevice(device)
            if hasattr(self.parent_app, 'log_status'):
                self.parent_app.log_status(f"Zvukový výstup změněn na: {device.description()}")

    def save_and_close(self):
        MODEL_CFG.whisper_language = self.combo_language.currentData()
        MODEL_CFG.device = self.cb_device.currentText()
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
        
        if sys.platform == "win32":
            try:
                hwnd = HWND(int(self.winId()))
                bg_colorref = 0x81b910  # #10b981
                txt_colorref = 0xffffff
                ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, DWORD(35), ctypes.byref(DWORD(bg_colorref)), ctypes.sizeof(DWORD))
                ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, DWORD(36), ctypes.byref(DWORD(txt_colorref)), ctypes.sizeof(DWORD))
            except: pass

        self.setStyleSheet("QDialog { background-color: #f9fafb; } QLabel { color: #1f2937; } QPushButton { background-color: #ffffff; border: 1px solid #d1d5db; border-radius: 4px; padding: 6px 15px; color: #4b5563; font-weight: 500; } QPushButton:hover { background-color: #e5e7eb; border: 1px solid #9ca3af; }")
        
        layout = QVBoxLayout(self)
        title = QLabel(f"Metoda: {res.method.upper()} | Skóre vzdálenosti: {res.score:.4f}")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #10b981; margin-bottom: 5px;")
        layout.addWidget(title, alignment=Qt.AlignmentFlag.AlignCenter)

        images_container = QHBoxLayout()
        frames_per_sec = res.sr / AUDIO_CFG.hop_length

        def add_black_padding(spec, padding_sec=0.15):
            pad_frames = int(padding_sec * frames_per_sec)
            return np.pad(spec, ((0, 0), (pad_frames, pad_frames)), mode='constant', constant_values=np.min(spec))

        spec_q = add_black_padding(res.vis_spec_query)
        dur_q = spec_q.shape[1] / frames_per_sec
        spec_m = add_black_padding(res.vis_spec_long[:, int(res.start_f):int(res.end_f)])
        dur_m = spec_m.shape[1] / frames_per_sec

        def create_spec_widget(spec, dur, title_text):
            fig = Figure(figsize=(0.6 + (dur / 10.0) * 5.0, 3.5), dpi=100, facecolor='#f9fafb')
            ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off')
            librosa.display.specshow(spec, ax=ax, cmap='magma', sr=res.sr, hop_length=AUDIO_CFG.hop_length)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), pad_inches=0); buf.seek(0)
            
            w = QWidget(); l = QVBoxLayout(w); l.setContentsMargins(0, 0, 0, 0)
            lbl_t = QLabel(title_text); lbl_t.setStyleSheet("color: #4b5563; font-size: 11px; font-weight: bold;")
            lbl_i = QLabel(); lbl_i.setPixmap(QPixmap.fromImage(QImage.fromData(buf.getvalue())))
            l.addWidget(lbl_t, alignment=Qt.AlignmentFlag.AlignCenter); l.addWidget(lbl_i, alignment=Qt.AlignmentFlag.AlignCenter)
            return w

        images_container.addStretch(1)
        images_container.addWidget(create_spec_widget(spec_q, dur_q, "Hledaný vzorek"))
        images_container.addWidget(create_spec_widget(spec_m, dur_m, f"Nalezený úsek (Čas: {res.start_f / frames_per_sec:.2f} s)"))
        images_container.addStretch(1)
        
        layout.addLayout(images_container)
        btn_close = QPushButton("Zavřít"); btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close, alignment=Qt.AlignmentFlag.AlignCenter)