# ==============================================================================
# audio_matcher/config.py
# ==============================================================================
import sys
import os
import json
from dataclasses import dataclass, asdict

if getattr(sys, 'frozen', False):
    BUNDLE_DIR = sys._MEIPASS 
    APP_DIR = os.path.dirname(sys.executable) 
else:
    BUNDLE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    APP_DIR = BUNDLE_DIR

SETTINGS_FILE = os.path.join(APP_DIR, "app_settings.json")
ASSETS_DIR = os.path.join(BUNDLE_DIR, "assets")

@dataclass
class AudioConfig:
    sr: int = 22050
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 64
    n_mfcc: int = 13
    fmin: int = 100
    trim_top_db: int = 20

@dataclass
class AppConfig:
    offline_db_path: str = os.path.join(APP_DIR, "data", "processed", "English_words_wav")
    database_file: str = os.path.join(APP_DIR, "data", "processed", "words_db.h5")
    
    icon_path: str = os.path.join(ASSETS_DIR, "ikona.ico")
    search_icon_path: str = os.path.join(ASSETS_DIR, "search.png")
    audio_icon_path: str = os.path.join(ASSETS_DIR, "audio.png")
    
    view_window_sec: float = 10.0
    min_speed_ratio: float = 0.5
    max_speed_ratio: float = 2.0
    supported_extensions: tuple = ("", ".wav", ".mp3", ".m4a", ".flac", ".ogg")
    
    whisper_min_confidence: float = 0.0
    hybrid_padding_sec: float = 0.5
    exclusion_padding_frames: int = 0
    dtw_max_retries: int = 10
    dtw_early_stop_threshold: float = 0.85 
    playback_padding_sec: float = 0.2

@dataclass
class AIModelConfig:
    whisper_language: str = "en"
    model_target_sr: int = 16000
    dtw_metric: str = "cosine"
    dtw_boundary_tolerance: float = 0.2
    dtw_penalty_value: float = 1000.0
    
    wav2vec_local_path: str = os.path.join(APP_DIR, "models", "wav2vec2-local")
    silero_vad_local_path: str = os.path.join(APP_DIR, "models", "silero_vad.jit")
    
    chunk_length_sec: float = 30.0
    chunk_overlap_sec: float = 2.0
    silero_vad_threshold: float = 0.5
    device: str = "cuda" 

@dataclass
class UIConfig:
    spectrogram_cmap: str = "magma"
    spectrogram_vmin: int = -80
    spectrogram_vmax: int = 0

AUDIO_CFG = AudioConfig()
APP_CFG = AppConfig()
MODEL_CFG = AIModelConfig()
UI_CFG = UIConfig()

def get_default_settings():
    return {
        "audio": asdict(AudioConfig()),
        "app": asdict(AppConfig()),
        "model": asdict(AIModelConfig()),
        "ui": asdict(UIConfig())
    }

def save_settings():
    data = {
        "audio": asdict(AUDIO_CFG),
        "app": asdict(APP_CFG),
        "model": asdict(MODEL_CFG),
        "ui": asdict(UI_CFG)
    }
    
    temp_file = SETTINGS_FILE + ".tmp"
    try:
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_file, SETTINGS_FILE)
    except Exception as e:
        print(f"Kritická chyba při ukládání nastavení: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)

def load_settings():
    if not os.path.exists(SETTINGS_FILE):
        print(f"Konfigurační soubor nebyl nalezen. Generuji výchozí: {SETTINGS_FILE}")
        data = get_default_settings()
        try:
            with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except Exception as e:
             print(f"Nepodařilo se vytvořit výchozí konfiguraci: {e}")
    else:
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if "audio" in data: AUDIO_CFG.__dict__.update(data["audio"])
            if "app" in data: APP_CFG.__dict__.update(data["app"])
            if "model" in data: MODEL_CFG.__dict__.update(data["model"])
            if "ui" in data: UI_CFG.__dict__.update(data["ui"])
            
            APP_CFG.supported_extensions = tuple(APP_CFG.supported_extensions)
        except Exception as e:
            print(f"Chyba při načítání nastavení: {e}")

load_settings()

MODERN_STYLESHEET = """
QMainWindow, QDialog { background-color: #f3f4f6; color: #1f2937; font-family: 'Segoe UI', -apple-system, Roboto, sans-serif; font-size: 9pt; }
QWidget { color: #1f2937; }
QFrame#cardPanel { background-color: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; }
QPushButton { background-color: #ffffff; border: 1px solid #d1d5db; border-radius: 6px; padding: 6px 16px; color: #374151; font-weight: 500; }
QPushButton:hover { background-color: #f9fafb; border-color: #9ca3af; color: #111827; }
QPushButton:pressed { background-color: #e5e7eb; }
QPushButton:disabled { background-color: #f3f4f6; border: 1px solid #e5e7eb; color: #9ca3af; }
QPushButton#primaryButton { background-color: #10b981; border: 1px solid #059669; color: white; font-weight: bold; padding: 8px 20px; }
QPushButton#primaryButton:hover { background-color: #059669; border-color: #047857; }
QPushButton#primaryButton:disabled { background-color: #a7f3d0; border-color: #6ee7b7; color: #ffffff; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background-color: #ffffff; border: 1px solid #d1d5db; border-radius: 6px; padding: 6px 12px; color: #1f2937; selection-background-color: #10b981; selection-color: white; }
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border: 1px solid #10b981; }
QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: top right; width: 28px; border-left: 1px solid #e5e7eb; border-top-right-radius: 5px; border-bottom-right-radius: 5px; background-color: #f9fafb; }
QComboBox::drop-down:hover { background-color: #f3f4f6; }
QComboBox::down-arrow { image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%236b7280' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'><polyline points='6 9 12 15 18 9'/></svg>"); width: 16px; height: 16px; }
QComboBox::down-arrow:on { image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%2310b981' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'><polyline points='18 15 12 9 6 15'/></svg>"); }
QComboBox QAbstractItemView { background-color: #ffffff; border: 1px solid #d1d5db; border-radius: 4px; selection-background-color: #10b981; outline: none; }
QComboBox QAbstractItemView::item { padding: 8px 12px; min-height: 24px; color: #1f2937; }
QComboBox QAbstractItemView::item:hover, QComboBox QAbstractItemView::item:selected { background-color: #10b981; color: #ffffff; }
QTabWidget::pane { border: none; border-top: 1px solid #e5e7eb; background-color: #f3f4f6; }
QTabBar::tab { background-color: transparent; padding: 10px 20px; color: #6b7280; border-bottom: 2px solid transparent; font-size: 10pt; font-weight: 500; }
QTabBar::tab:hover { color: #1f2937; }
QTabBar::tab:selected { color: #10b981; border-bottom: 2px solid #10b981; }
QProgressBar { border: none; background-color: #e5e7eb; border-radius: 3px; height: 6px; }
QProgressBar::chunk { background-color: #10b981; border-radius: 3px; }
QTextEdit#logConsole { background-color: #ffffff; border: 1px solid #e5e7eb; border-radius: 6px; color: #374151; font-family: 'Consolas', 'Courier New', monospace; font-size: 8pt; padding: 8px; }
QTableWidget { background-color: #ffffff; color: #1f2937; gridline-color: #e5e7eb; border: 1px solid #d1d5db; border-radius: 6px; selection-background-color: #10b981; selection-color: #ffffff; outline: none; }
QTableWidget::item { padding: 4px 8px; border-bottom: 1px solid #f9fafb; }
QTableWidget::item:selected { background-color: #10b981; color: #ffffff; }
QHeaderView::section { background-color: #f3f4f6; color: #4b5563; padding: 6px 8px; border: none; border-right: 1px solid #e5e7eb; border-bottom: 1px solid #d1d5db; font-weight: 600; }
QHeaderView::section:last { border-right: none; }
QTableCornerButton::section { background-color: #f3f4f6; border: none; border-bottom: 1px solid #d1d5db; }
QScrollBar:vertical { border: none; background: #f3f4f6; width: 10px; margin: 2px; border-radius: 4px; }
QScrollBar::handle:vertical { background: #d1d5db; min-height: 25px; border-radius: 4px; }
QScrollBar::handle:vertical:hover { background: #9ca3af; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { border: none; background: none; height: 0px; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
"""

COMMON_STYLE_INPUTS = """
    QComboBox, QLineEdit { padding: 6px 10px; border: 1px solid #d1d5db; border-radius: 4px; background-color: white; font-size: 10pt; }
    QComboBox:focus, QLineEdit:focus { border: 1px solid #10b981; }
"""

COMMON_STYLE_BUTTONS = """
    QPushButton { background-color: #f3f4f6; border: 1px solid #d1d5db; border-radius: 4px; padding: 6px 15px; color: #4b5563; font-weight: 500; font-size: 10pt; }
    QPushButton:hover { background-color: #e5e7eb; border: 1px solid #9ca3af; }
"""