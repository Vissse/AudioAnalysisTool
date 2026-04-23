import os
import sys
import tempfile
import ctypes
import numpy as np
import soundfile as sf
import winsound
import csv
from datetime import datetime
from ctypes.wintypes import HWND, DWORD

from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QComboBox,
                             QTextEdit, QMessageBox, QLineEdit, QTabWidget,
                             QTableWidget, QTableWidgetItem, QHeaderView,
                             QProgressBar, QSplitter, QFrame, QApplication,
                             QInputDialog)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtGui import QIcon, QBrush, QColor

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import librosa
import librosa.display

from config import APP_CFG, AUDIO_CFG, MODERN_STYLESHEET, UI_CFG
from data_types import MatchResult
from gui.workers import SingleAnalysisWorker, CorpusScannerWorker, BatchEvaluationWorker
from gui.dialogs import VisualCompareDialog, SettingsDialog
from core.audio_utils import load_and_prep


class AudioMatcherApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Analysis Tool")
        self.resize(1300, 950) 
        self.setStyleSheet(MODERN_STYLESHEET)
        
        # Audio Engine
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.audio_output.setVolume(1.0)
        self.audio_output.setMuted(False)
        self.player.setAudioOutput(self.audio_output)

        # Data State
        self.single_found_ranges = []
        self.single_current_result: MatchResult | None = None
        self.corpus_data = None
        self.corpus_history = {}
        self.corpus_current_result: MatchResult | None = None
        
        # Paměť pro kvantitativní test
        self.batch_long_path = None
        self.batch_query_path = None
        self.batch_detailed_data = {} 
        self.ground_truth_path = None 
        
        self.init_ui()
        self.apply_native_titlebar_color()
        self.center_window()

    def log_status(self, message: str):
        """Vypíše zprávu do konzole a okamžitě donutí grafické rozhraní k překreslení."""
        self.log_text.append(f"{message}")
        QApplication.processEvents()
    
    def center_window(self):
        """Vycentruje okno aplikace přesně doprostřed aktuálního monitoru."""
        qr = self.frameGeometry() # Získá aktuální geometrii okna (včetně lišty)
        cp = self.screen().availableGeometry().center() # Najde střed obrazovky
        qr.moveCenter(cp) # Přesune pomyslný obdélník okna na střed obrazovky
        self.move(qr.topLeft()) # Skutečně přesune okno aplikace na tyto nové souřadnice
        
    def init_ui(self):
        main_widget = QWidget()
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        
        self.setWindowIcon(QIcon(APP_CFG.icon_path))

        self.tabs = QTabWidget()
        self.tabs.tabBar().setCursor(Qt.CursorShape.PointingHandCursor)

        self.tabs.setStyleSheet("""
            QTabBar::tab {
                border: none;
                border-right: 1px solid #e5e7eb;
                padding: 10px 20px;
                color: #6b7280;
                background: transparent;
                font-weight: 500;
            }
            QTabBar::tab:last { border-right: none; }
            QTabBar::tab:hover { color: #10b981; background-color: #f9fafb; }
            QTabBar::tab:selected {
                color: #10b981;
                border-bottom: 2px solid #10b981;
                font-weight: bold;
            }
            QTabWidget::pane {
                border: none;
                border-top: 1px solid #e5e7eb; 
            }
        """)

        self.btn_settings = QPushButton("Nastavení")
        # Změna kurzoru na ručičku při najetí
        self.btn_settings.setCursor(Qt.CursorShape.PointingHandCursor) 
        self.btn_settings.setStyleSheet("""
            QPushButton { 
                background: transparent; 
                border: none; 
                border-bottom: 2px solid transparent; 
                border-radius: 0px; 
                font-size: 10pt;
                font-weight: 500; 
                color: #6b7280; 
                padding: 10px 20px; 
            }
            QPushButton:hover { 
                color: #10b981; 
                background-color: #f9fafb; 
                border-radius: 0px;
            }
            QPushButton:pressed { 
                color: #10b981; 
                border-bottom: 2px solid #10b981; 
                font-weight: bold; 
                border-radius: 0px;
            }
        """)
        self.btn_settings.clicked.connect(self.open_settings)
        self.tabs.setCornerWidget(self.btn_settings, Qt.Corner.TopRightCorner)

        self.tab_detail = QWidget()
        self.setup_detail_tab()
        
        self.tab_corpus = QWidget()
        self.setup_corpus_tab()
        
        self.tab_batch = QWidget()
        self.setup_batch_tab()
        
        self.tabs.addTab(self.tab_detail, "Srovnávací analýza")
        self.tabs.addTab(self.tab_corpus, "Analýza korpusu")
        self.tabs.addTab(self.tab_batch, "Kvantitativní evaluace")
        
        layout.addWidget(self.tabs)
        self.setCentralWidget(main_widget)

    # ==========================================
    # 🎨 TAB 1: Srovnávací analýza
    # ==========================================
    def setup_detail_tab(self):
        layout = QVBoxLayout(self.tab_detail)
        layout.setSpacing(15) 
        layout.setContentsMargins(20, 20, 20, 20)
        
        # --- HLAVNÍ KARTA ---
        input_frame = QFrame()
        input_frame.setObjectName("cardPanel")
        input_layout = QVBoxLayout(input_frame)
        input_layout.setSpacing(15)
        input_layout.setContentsMargins(20, 20, 20, 20)

        # --- NADPIS A POPIS ---
        title_detail = QLabel("Srovnávací analýza")
        title_detail.setStyleSheet("font-size: 14pt; font-weight: bold;")
        desc_detail = QLabel("Algoritmus porovná hledaný vzorek s prohledávaným audiem a detailně vizualizuje shodu.")
        desc_detail.setStyleSheet("color: #6b7280; margin-bottom: 10px;")
        
        input_layout.addWidget(title_detail)
        input_layout.addWidget(desc_detail)

        # -------- DEFINICE STYLŮ --------
        standard_input_style = """
            QComboBox, QLineEdit {
                padding: 6px 10px;
                border: 1px solid #d1d5db;
                border-radius: 4px;
                background-color: white;
                font-size: 10pt;
            }
            QComboBox:focus, QLineEdit:focus {
                border: 1px solid #10b981;
            }
        """

        browse_btn_style = """
            QPushButton {
                background-color: #f3f4f6;
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
        """

        INPUT_WIDTH = 375
        LABEL_WIDTH = 130

        def add_search_icon_to_lineedit(line_edit: QLineEdit):
            from PyQt6.QtGui import QIcon
            search_icon = QIcon('assets/search.png')
            search_action = line_edit.addAction(search_icon, QLineEdit.ActionPosition.TrailingPosition)
            line_edit.setStyleSheet(f"QLineEdit {{ padding: 6px 10px; border: 1px solid #d1d5db; border-radius: 4px; background-color: white; font-size: 10pt; }} QLineEdit:focus {{ border: 1px solid #10b981; }}")
            search_action.triggered.connect(line_edit.returnPressed)
            return line_edit

        # -------- ROW 1: Prohledávané Audio --------
        row1 = QHBoxLayout()
        lbl_long_text = QLabel("Prohledávané Audio:")
        lbl_long_text.setFixedWidth(LABEL_WIDTH)
        
        self.input_single_long = QLineEdit()
        self.input_single_long.setPlaceholderText("Hledejte audio v DB (zadejte číslo od 0 do 4075)")
        self.input_single_long.setFixedWidth(INPUT_WIDTH)
        add_search_icon_to_lineedit(self.input_single_long)
        self.input_single_long.returnPressed.connect(self.load_single_long_from_folder)
        
        self.btn_single_browse_long = QPushButton("Procházet vlastní...")
        self.btn_single_browse_long.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_single_browse_long.setStyleSheet(browse_btn_style)
        self.btn_single_browse_long.clicked.connect(lambda: self.select_file('long_single'))
        
        self.lbl_single_long = QLabel("Nebylo vybráno")
        self.lbl_single_long.setStyleSheet("color: #6b7280; font-style: italic; font-size: 9pt;")
        
        row1.addWidget(lbl_long_text)
        row1.addWidget(self.input_single_long)
        row1.addSpacing(10)
        row1.addWidget(self.btn_single_browse_long)
        row1.addWidget(self.lbl_single_long)
        row1.addStretch()
        input_layout.addLayout(row1)

        # -------- ROW 2: Hledaný Vzorek --------
        row2 = QHBoxLayout()
        lbl_query_text = QLabel("Hledaný Vzorek:")
        lbl_query_text.setFixedWidth(LABEL_WIDTH)
        
        self.input_query = QLineEdit()
        self.input_query.setPlaceholderText("Hledejte vzorek v DB (např. 'people', ...)")
        self.input_query.setFixedWidth(INPUT_WIDTH)
        add_search_icon_to_lineedit(self.input_query)
        self.input_query.returnPressed.connect(self.load_query_from_folder)
        
        self.btn_browse_query = QPushButton("Procházet vlastní...")
        self.btn_browse_query.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_browse_query.setStyleSheet(browse_btn_style)
        self.btn_browse_query.clicked.connect(lambda: self.select_file('query_single'))
        
        self.lbl_single_query = QLabel("Nebylo vybráno")
        self.lbl_single_query.setStyleSheet("color: #6b7280; font-style: italic; font-size: 9pt;")
        
        row2.addWidget(lbl_query_text)
        row2.addWidget(self.input_query)
        row2.addSpacing(10)
        row2.addWidget(self.btn_browse_query)
        row2.addWidget(self.lbl_single_query)
        row2.addStretch()
        input_layout.addLayout(row2)

        # -------- ROW 3: Model --------
        row3 = QHBoxLayout()
        lbl_model_text = QLabel("Analytický Model:")
        lbl_model_text.setFixedWidth(LABEL_WIDTH)
        
        self.combo_method = QComboBox()
        self.combo_method.setCursor(Qt.CursorShape.PointingHandCursor)
        self.combo_method.addItems(["OpenAI Whisper (ASR)", "Wav2Vec 2.0 + DTW", "MFCC + DTW", "Pattern Matching"])
        self.combo_method.setItemData(0, "whisper"); self.combo_method.setItemData(1, "wav2vec")
        self.combo_method.setItemData(2, "dtw"); self.combo_method.setItemData(3, "pattern")
        self.combo_method.setFixedWidth(INPUT_WIDTH)
        self.combo_method.setStyleSheet(standard_input_style)
        
        row3.addWidget(lbl_model_text)
        row3.addWidget(self.combo_method)
        row3.addStretch()
        input_layout.addLayout(row3)

        # -------- SEPARÁTOR --------
        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine); sep.setFrameShadow(QFrame.Shadow.Sunken)
        sep.setStyleSheet("background-color: #e5e7eb; margin: 5px 0;")
        input_layout.addWidget(sep)

       # --- SEKCE TLAČÍTEK ---
        controls_layout = QHBoxLayout()
        self.btn_analyze = QPushButton("SPUSTIT ANALÝZU")
        self.btn_analyze.setObjectName("primaryButton")
        self.btn_analyze.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_analyze.setMinimumWidth(180)
        self.btn_analyze.clicked.connect(self.run_single_analysis)
        
        self.btn_next = QPushButton("Zobrazit další nález")
        self.btn_next.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_next.setStyleSheet(browse_btn_style)
        self.btn_next.clicked.connect(self.run_single_next)
        self.btn_next.setEnabled(False)

        # Tlačítko přesunuto sem vedle "Zobrazit další nález"
        self.btn_visual = QPushButton("Vizuální srovnání")
        self.btn_visual.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_visual.setStyleSheet(browse_btn_style)
        self.btn_visual.clicked.connect(lambda: self.open_visual_comparison(self.single_current_result))
        self.btn_visual.setEnabled(False)

        controls_layout.addWidget(self.btn_analyze)
        controls_layout.addWidget(self.btn_next)
        controls_layout.addWidget(self.btn_visual)
        controls_layout.addStretch() # Natažení mezery až za vizuálním srovnáním
        
        self.btn_listen_L, self.btn_listen_stereo, self.btn_listen_R = self.create_listen_buttons(self.play_single)
        for btn in [self.btn_listen_L, self.btn_listen_stereo, self.btn_listen_R]:
            btn.setStyleSheet(browse_btn_style)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setEnabled(False) # Zablokování tlačítek pro přehrávání po startu
            controls_layout.addWidget(btn)
            
        input_layout.addLayout(controls_layout)

        # --- STAV ---
        self.progress_detail = QProgressBar(); self.progress_detail.setFixedHeight(6); self.progress_detail.setTextVisible(False)
        input_layout.addWidget(self.progress_detail)
        self.log_text = QTextEdit(); self.log_text.setObjectName("logConsole"); self.log_text.setMaximumHeight(80)
        input_layout.addWidget(self.log_text)

        layout.addWidget(input_frame)
        self.canvas_single = FigureCanvas(Figure(figsize=(12, 5), dpi=100))
        self.canvas_single.figure.patch.set_facecolor('#f3f4f6') 
        layout.addWidget(self.canvas_single, 1)

    # ==========================================
    # 🎨 TAB 2: ANALÝZA KORPUSU
    # ==========================================
    def setup_corpus_tab(self):
        layout = QVBoxLayout(self.tab_corpus)
        layout.setSpacing(15) # Sjednocení vnějších mezer
        layout.setContentsMargins(20, 20, 20, 20)
        
        input_frame = QFrame()
        input_frame.setObjectName("cardPanel")
        
        # Změněno z QHBoxLayout na QVBoxLayout a přidány stejné okraje
        input_layout = QVBoxLayout(input_frame)
        input_layout.setSpacing(15)
        input_layout.setContentsMargins(20, 20, 20, 20)

        # --- NADPIS A POPIS ---
        title_corpus = QLabel("Analýza korpusu")
        title_corpus.setStyleSheet("font-size: 14pt; font-weight: bold;")
        desc_corpus = QLabel("Nástroj pro hromadné prohledání nahrávky a detekci všech výskytů podle zvoleného analytického modelu.")
        desc_corpus.setStyleSheet("color: #6b7280; margin-bottom: 10px;")
        
        input_layout.addWidget(title_corpus)
        input_layout.addWidget(desc_corpus)
        
        # --- OVLÁDACÍ PRVKY V ŘÁDKU ---
        controls_layout = QHBoxLayout()
        
        self.btn_corpus_long = QPushButton("Vybrat delší nahrávku")
        self.lbl_corpus_long = QLabel("Nebylo vybráno")
        self.btn_corpus_long.clicked.connect(lambda: self.select_file('long_corpus'))
        
        self.combo_corpus_method = QComboBox()
        self.combo_corpus_method.setCursor(Qt.CursorShape.PointingHandCursor)
        self.combo_corpus_method.addItems([
            "OpenAI Whisper (ASR)", 
            "Whisper + DTW Hybrid",
            "Wav2Vec 2.0 + DTW", 
            "MFCC + DTW", 
            "Pattern Matching"
        ])
        self.combo_corpus_method.setItemData(0, "whisper")
        self.combo_corpus_method.setItemData(1, "whisper_hybrid")
        self.combo_corpus_method.setItemData(2, "wav2vec")
        self.combo_corpus_method.setItemData(3, "dtw")
        self.combo_corpus_method.setItemData(4, "pattern")
        
        self.btn_corpus_scan = QPushButton("ANALYZOVAT KORPUS")
        self.btn_corpus_scan.setObjectName("primaryButton")
        self.btn_corpus_scan.clicked.connect(self.run_corpus_scan)

        # NOVÉ: Tlačítko pro pauzu
        self.btn_corpus_pause = QPushButton("Pozastavení analýzy")
        self.btn_corpus_pause.setEnabled(False) # Aktivuje se až při běhu
        self.btn_corpus_pause.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_corpus_pause.clicked.connect(self.toggle_corpus_pause)
        
        # Propojení změny modelu se schováním textu
        self.combo_corpus_method.currentIndexChanged.connect(self.on_corpus_method_changed)
        
        controls_layout.addWidget(self.btn_corpus_long)
        controls_layout.addWidget(self.lbl_corpus_long)
        controls_layout.addStretch(1)
        controls_layout.addWidget(self.combo_corpus_method)
        controls_layout.addWidget(self.btn_corpus_pause) # Přidáno do layoutu
        controls_layout.addWidget(self.btn_corpus_scan)

        # Celý řádek přidáme pod nadpis
        input_layout.addLayout(controls_layout)

        self.progress_corpus = QProgressBar()
        self.progress_corpus.setFixedHeight(6)
        self.progress_corpus.setTextVisible(False)
        
        self.log_corpus = QTextEdit()
        self.log_corpus.setReadOnly(True)
        self.log_corpus.setObjectName("logConsole")
        self.log_corpus.setMaximumHeight(80)
        self.log_corpus.setPlaceholderText("Zde se bude vypisovat postup skenování korpusu...")

        # --- Rámeček pro zobrazení textu nahrávky ---
        self.frame_corpus_gt = QFrame()
        self.frame_corpus_gt.setStyleSheet("""
            QFrame {
                background-color: #ecfdf5; 
                border: 1px solid #6ee7b7; 
                border-radius: 6px; 
            }
        """)
        gt_layout = QVBoxLayout(self.frame_corpus_gt)
        gt_layout.setContentsMargins(10, 10, 10, 10)
        gt_layout.setSpacing(6) # <- Zde se teď nastavuje mezera mezi prvním a druhým textem
        
        # Jednotný styl pro oba texty (garantuje 0 pixelů odskok)
        common_style = "color: #065f46; background: transparent; border: none; font-size: 10pt; margin: 0px; padding: 0px;"
        
        self.lbl_corpus_gt = QLabel()
        self.lbl_corpus_gt.setWordWrap(True)
        self.lbl_corpus_gt.setStyleSheet(common_style)
        gt_layout.addWidget(self.lbl_corpus_gt)
        
        self.lbl_whisper_text = QLabel()
        self.lbl_whisper_text.setWordWrap(True)
        self.lbl_whisper_text.setStyleSheet(common_style)
        self.lbl_whisper_text.hide()
        self.lbl_whisper_text.setText("")
        gt_layout.addWidget(self.lbl_whisper_text)
        
        self.frame_corpus_gt.hide()
        # --------------------------------------------------
        # --------------------------------------------------

        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Slovo", "Skóre", "Čas (s)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setDefaultSectionSize(40) 
        
        # NOVÉ: Skrytí šedého pruhu u prázdných řádků
        self.table.setStyleSheet("QHeaderView { background-color: #ffffff; border: none; }")
        
        self.canvas_corpus = FigureCanvas(Figure(figsize=(8, 5), dpi=100))
        self.canvas_corpus.figure.patch.set_facecolor('#f3f4f6') 

        layout.addWidget(input_frame)
        layout.addWidget(self.progress_corpus)
        layout.addWidget(self.log_corpus)
        layout.addWidget(self.frame_corpus_gt) # Vložen rámeček nad tabulku
        layout.addWidget(self.table, 4)
        layout.addWidget(self.canvas_corpus, 1)

    # ==========================================
    # 📈 TAB 3: KVANTITATIVNÍ EVALUACE
    # ==========================================
    def setup_batch_tab(self):
        layout = QVBoxLayout(self.tab_batch)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        batch_frame = QFrame()
        batch_frame.setObjectName("cardPanel")
        batch_layout = QVBoxLayout(batch_frame)

        batch_layout.setSpacing(15)
        batch_layout.setContentsMargins(20, 20, 20, 20)
        
        title_batch = QLabel("Kvantitativní evaluace")
        title_batch.setStyleSheet("font-size: 14pt; font-weight: bold;")
        desc_batch = QLabel("Algoritmus projde nahrávku, najde všechny výskyty zadaného slova a vygeneruje statistiku vůči zlatému standardu.")
        desc_batch.setStyleSheet("color: #6b7280; margin-bottom: 10px;")

        batch_controls = QVBoxLayout()
        batch_controls.setSpacing(10)
        
        # -------- DEFINICE STYLŮ (Zmenšený font a upravený padding) --------
        standard_input_style = """
            QComboBox, QLineEdit {
                padding: 4px 6px; /* Zmenšené odsazení, aby text začínal dříve */
                border: 1px solid #d1d5db;
                border-radius: 4px;
                background-color: white;
                font-size: 10pt; /* Zmenšený font */
            }
            QComboBox:focus, QLineEdit:focus {
                border: 1px solid #10b981;
            }
            QComboBox::item { padding: 5px; }
        """

        browse_btn_style = """
            QPushButton {
                background-color: #f3f4f6;
                border: 1px solid #d1d5db;
                border-radius: 4px;
                padding: 5px 12px;
                color: #4b5563;
                font-weight: 500;
                font-size: 10pt;
            }
            QPushButton:hover {
                background-color: #e5e7eb;
                border: 1px solid #9ca3af;
            }
        """

        INPUT_WIDTH = 375
        LABEL_WIDTH = 130

        # Pomocná funkce pro lupu (s menším fontem)
        def add_search_icon_to_lineedit(line_edit: QLineEdit):
            search_icon = QIcon('assets/search.png')
            search_action = line_edit.addAction(search_icon, QLineEdit.ActionPosition.TrailingPosition)
            line_edit.setStyleSheet(f"QLineEdit {{ padding: 4px 6px; border: 1px solid #d1d5db; border-radius: 4px; background-color: white; font-size: 10pt; }} QLineEdit:focus {{ border: 1px solid #10b981; }}")
            search_action.triggered.connect(line_edit.returnPressed)
            return line_edit

        # -------- ROW 0: Zlatý standard --------
        row0 = QHBoxLayout()
        lbl_gt = QLabel("Zlatý standard:")
        lbl_gt.setFixedWidth(LABEL_WIDTH)
        
        self.combo_gt = QComboBox()
        self.combo_gt.setEditable(True)
        self.combo_gt.addItems([
            "virtual_stream_ground_truth_complete_time.json",
            "virtual_stream_ground_truth.json"
        ])
        self.combo_gt.setFixedWidth(INPUT_WIDTH)
        self.combo_gt.setStyleSheet(standard_input_style)
        
        self.btn_select_gt = QPushButton("Procházet vlastní...")
        self.btn_select_gt.setStyleSheet(browse_btn_style)
        self.btn_select_gt.clicked.connect(self.select_gt_file)
        
        row0.addWidget(lbl_gt)
        row0.addWidget(self.combo_gt)
        row0.addSpacing(15)
        row0.addWidget(self.btn_select_gt)
        row0.addStretch()
        batch_controls.addLayout(row0)
        
        # -------- ROW 1: Prohledávané Audio --------
        row1 = QHBoxLayout()
        lbl_long = QLabel("Prohledávané audio:")
        lbl_long.setFixedWidth(LABEL_WIDTH)
        self.input_batch_long = QLineEdit()
        self.input_batch_long.setPlaceholderText("Hledejte audio v DB (zadejte číslo od 0 do 4075)")
        self.input_batch_long.setFixedWidth(INPUT_WIDTH)
        add_search_icon_to_lineedit(self.input_batch_long)
        self.input_batch_long.returnPressed.connect(self.load_batch_long_from_folder)
        
        self.btn_batch_browse_long = QPushButton("Procházet vlastní...")
        self.btn_batch_browse_long.setStyleSheet(browse_btn_style)
        self.btn_batch_browse_long.clicked.connect(self.select_batch_long)
        
        self.lbl_batch_long = QLabel("Nebylo vybráno")
        self.lbl_batch_long.setStyleSheet("color: #6b7280; font-style: italic; font-size: 9pt;")
        
        row1.addWidget(lbl_long)
        row1.addWidget(self.input_batch_long)
        row1.addSpacing(15)
        row1.addWidget(self.btn_batch_browse_long)
        row1.addWidget(self.lbl_batch_long)
        row1.addStretch()
        batch_controls.addLayout(row1)
        
        # -------- ROW 2: Hledaný vzorek --------
        row2 = QHBoxLayout()
        lbl_query = QLabel("Hledaný vzorek:")
        lbl_query.setFixedWidth(LABEL_WIDTH)
        self.input_batch_query = QLineEdit()
        self.input_batch_query.setPlaceholderText("Hledejte vzorek v DB (např. 'people', ...)")
        self.input_batch_query.setFixedWidth(INPUT_WIDTH)
        add_search_icon_to_lineedit(self.input_batch_query)
        self.input_batch_query.returnPressed.connect(self.load_batch_query_from_folder)
        
        self.btn_batch_query = QPushButton("Procházet vlastní...")
        self.btn_batch_query.setStyleSheet(browse_btn_style)
        self.btn_batch_query.clicked.connect(self.select_batch_query)
        
        self.lbl_batch_query = QLabel("Nebylo vybráno")
        self.lbl_batch_query.setStyleSheet("color: #6b7280; font-style: italic; font-size: 9pt;")
        
        row2.addWidget(lbl_query)
        row2.addWidget(self.input_batch_query)
        row2.addSpacing(15)
        row2.addWidget(self.btn_batch_query)
        row2.addWidget(self.lbl_batch_query)
        row2.addStretch()
        batch_controls.addLayout(row2)

        # -------- ROW 3: Model a Práh (Práh posunut vpravo) --------
        row3 = QHBoxLayout()
        lbl_model_text = QLabel("Analytický model:")
        lbl_model_text.setFixedWidth(LABEL_WIDTH)
        
        self.combo_batch_method = QComboBox()
        self.combo_batch_method.addItems(["OpenAI Whisper (ASR)", "Wav2Vec 2.0 + DTW", "MFCC + DTW", "Pattern Matching"])
        self.combo_batch_method.setItemData(0, "whisper"); self.combo_batch_method.setItemData(1, "wav2vec")
        self.combo_batch_method.setItemData(2, "dtw"); self.combo_batch_method.setItemData(3, "pattern")
        self.combo_batch_method.setFixedWidth(INPUT_WIDTH)
        self.combo_batch_method.setStyleSheet(standard_input_style)
        
        self.btn_batch_model_browse = QPushButton("Procházet vlastní...")
        self.btn_batch_model_browse.setStyleSheet(browse_btn_style)
        # Zde můžete napojit funkci pro výběr vlastního modelu (.pt nebo složky)
        
        row3.addWidget(lbl_model_text)
        row3.addWidget(self.combo_batch_method)
        row3.addSpacing(15)
        row3.addWidget(self.btn_batch_model_browse)
        
        row3.addStretch() # Toto odsune práh na pravý konec
        
        lbl_threshold = QLabel("Maximální práh:")
        self.input_threshold = QLineEdit("1.0") 
        self.input_threshold.setFixedWidth(50)
        self.input_threshold.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.input_threshold.setStyleSheet(standard_input_style)
        
        row3.addWidget(lbl_threshold)
        row3.addWidget(self.input_threshold)
        batch_controls.addLayout(row3)
        # ----------------------------------------

        self.btn_run_batch = QPushButton("SPUSTIT KVANTITATIVNÍ EVALUACI")

        self.btn_run_batch.setObjectName("primaryButton")
        self.btn_run_batch.clicked.connect(self.run_batch_benchmark)

        self.progress_batch = QProgressBar()
        self.progress_batch.setFixedHeight(6)
        self.progress_batch.setTextVisible(False)
        self.lbl_batch_status = QLabel("Připraveno.")
        
        self.log_batch = QTextEdit()
        self.log_batch.setReadOnly(True)
        self.log_batch.setObjectName("logConsole")
        self.log_batch.setMaximumHeight(120)
        self.log_batch.setPlaceholderText("Zde se bude v reálném čase vypisovat postup algoritmu...")

        self.table_batch = QTableWidget()
        self.table_batch.setColumnCount(6)
        self.table_batch.setHorizontalHeaderLabels(["Slovo", "GT", "Nález", "FRR (%)", "FA/h", "F1"])
        self.table_batch.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_batch.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table_batch.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table_batch.itemSelectionChanged.connect(self.on_batch_row_selected) 
        
        self.table_batch.verticalHeader().setVisible(True) 
        self.table_batch.verticalHeader().setDefaultSectionSize(40) 

        self.table_batch.setStyleSheet("""
            QTableWidget {
                background-color: #ffffff; 
                alternate-background-color: #ffffff;
                border: none;
            }
            QHeaderView::section:vertical {
                background-color: #f9fafb; 
                color: #6b7280; 
                border: none;
                border-right: 1px solid #e5e7eb; 
                border-bottom: 1px solid #e5e7eb;
                padding-left: 8px;
                padding-right: 8px;
                font-weight: bold;
            }
            QTableCornerButton::section {
                background-color: #f9fafb;
                border: none;
                border-right: 1px solid #e5e7eb;
                border-bottom: 1px solid #e5e7eb;
            }
        """)

        self.txt_batch_detail = QTextEdit()
        self.txt_batch_detail.setReadOnly(True)
        self.txt_batch_detail.setObjectName("logConsole")
        self.txt_batch_detail.setPlaceholderText("Klikněte na slovo v tabulce vlevo pro zobrazení přesných časů...")

        batch_splitter = QSplitter(Qt.Orientation.Horizontal)
        batch_splitter.addWidget(self.table_batch)
        batch_splitter.addWidget(self.txt_batch_detail)
        batch_splitter.setSizes([600, 300]) 
        
        batch_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #10b981; 
                width: 2px; 
            }
            QSplitter::handle:horizontal {
                image: none;
            }
        """)

        batch_layout.addWidget(title_batch)
        batch_layout.addWidget(desc_batch)
        batch_layout.addLayout(batch_controls)
        batch_layout.addSpacing(10)
        batch_layout.addWidget(self.btn_run_batch)
        batch_layout.addWidget(self.progress_batch)
        batch_layout.addWidget(self.lbl_batch_status)
        batch_layout.addWidget(self.log_batch)
        batch_layout.addWidget(batch_splitter, 1)

        layout.addWidget(batch_frame, 1)

    # ==========================================
    # ⚙️ LOGIKA A OBSLUHA
    # ==========================================

    def display_ground_truth_text(self, audio_path):
        """Vyhledá referenční text nahrávky v JSONu a zobrazí ho nad tabulkou."""
        import json
        filename = os.path.basename(audio_path)
        filename_no_ext = os.path.splitext(filename)[0]
        
        # Hledáme na více místech (ve složce s DB, v nadřazené složce, i v kořenové složce aplikace)
        base_dir = os.path.dirname(APP_CFG.offline_db_path)
        root_dir = os.path.dirname(base_dir)
        
        gt_filenames = [
            "virtual_stream_ground_truth.json"
        ]
        
        gt_data = None
        loaded_path = ""
        
        # Zkusíme projít všechny kombinace cest
        for d in [base_dir, root_dir, os.getcwd()]:
            for f_name in gt_filenames:
                p = os.path.join(d, f_name)
                if os.path.exists(p):
                    try:
                        with open(p, 'r', encoding='utf-8') as f:
                            gt_data = json.load(f)
                            loaded_path = p
                        break
                    except Exception:
                        pass
            if gt_data:
                break
                
        if not gt_data:
            self.log_corpus.append("ℹ️ Poznámka: Nepodařilo se najít JSON soubor se zlatým standardem.")
            self.frame_corpus_gt.hide()
            return
            
        matching_words = []
        for w in gt_data:
            # Ochrana pro případ, že struktura JSONu není přesně slovník
            if not isinstance(w, dict):
                continue
                
            src = w.get('source_file') or w.get('filename') or ""
            
            # Bezpečné porovnání (bude fungovat i když je v JSONu cesta např. "cv_16k_wav/sample-000008.wav")
            if filename in src or filename_no_ext in src:
                # Preferujeme orthographic, ale máme záložní varianty
                text = w.get('orthographic') or w.get('word') or w.get('text')
                start_time = w.get('start') or w.get('stream_start') or 0.0
                
                if text:
                    matching_words.append({
                        "text": str(text),
                        "start": float(start_time)
                    })
                    
        if matching_words:
            # Seřadíme slova chronologicky podle času, aby věta dávala smysl
            matching_words.sort(key=lambda x: x["start"])
            
            # Vyčistíme mezery a pospojujeme do finální věty
            words_list = [item["text"].strip() for item in matching_words]
            full_text = " ".join(words_list)
            
            self.lbl_corpus_gt.setText(f"<b>Přepis nahrávky '{filename}' (Zlatý standard):</b> {full_text}")
            self.frame_corpus_gt.show()
            self.log_corpus.append(f"✅ Načten a složen referenční text ze souboru: {os.path.basename(loaded_path)}")
        else:
            self.log_corpus.append(f"ℹ️ Poznámka: V JSONu ({os.path.basename(loaded_path)}) nebyl nalezen žádný text pro nahrávku '{filename}'.")
            self.frame_corpus_gt.hide()
    
    
    def select_gt_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Vyberte zlatý standard", "", "JSON (*.json)")
        if path:
            self.combo_gt.setCurrentText(path)
            self.log_status(f"✅ Zlatý standard ručně vybrán: {os.path.basename(path)}")

    def select_batch_long(self):
        path, _ = QFileDialog.getOpenFileName(self, "Vyberte dlouhé audio pro test", "", "Audio (*.mp3 *.wav *.m4a *.flac)")
        if path:
            self.batch_long_path = path
            self.lbl_batch_long.setText(os.path.basename(path))
            self.lbl_batch_long.setStyleSheet("color: #10b981; font-weight: bold;")
            self.input_batch_long.setText(os.path.splitext(os.path.basename(path))[0])

    def load_batch_long_from_folder(self):
        audio_name = self.input_batch_long.text().strip()
        if not audio_name:
            return
            
        # Chytré doplnění čísla na formát sample-XXXXXX
        if audio_name.isdigit():
            audio_name = f"sample-{int(audio_name):06d}"
            self.input_batch_long.setText(audio_name) # Přepíše pole na správný formát
            
        # Aplikace chytře prohledá známé složky s daty
        base_dir = os.path.dirname(APP_CFG.offline_db_path)
        possible_dirs = [
            APP_CFG.offline_db_path,
            base_dir,
            os.path.join(base_dir, "cv_16k_wav"),
            os.path.join(base_dir, "Common Voice Spontaneous Speech 3.0 - English")
        ]
            
        # Aplikace chytře prohledá známé složky s daty
        base_dir = os.path.dirname(APP_CFG.offline_db_path)
        possible_dirs = [
            APP_CFG.offline_db_path,
            base_dir,
            os.path.join(base_dir, "cv_16k_wav"),
            os.path.join(base_dir, "Common Voice Spontaneous Speech 3.0 - English")
        ]
        
        for d in possible_dirs:
            for ext in APP_CFG.supported_extensions:
                p = os.path.join(d, audio_name + ext)
                if os.path.exists(p):
                    self.batch_long_path = p
                    self.lbl_batch_long.setText(os.path.basename(p))
                    self.lbl_batch_long.setStyleSheet("color: #10b981; font-weight: bold;")
                    self.log_status(f"✅ Audio korpusu '{audio_name}' načteno z DB.")
                    return
                    
        QMessageBox.warning(self, "Chyba", f"Audio '{audio_name}' nebylo v lokální databázi nalezeno.\n(Zkontrolujte název nebo použijte 'Procházet...')")

    def load_batch_query_from_folder(self):
        word = self.input_batch_query.text().strip()
        if not word:
            return
            
        for ext in APP_CFG.supported_extensions:
            p = os.path.join(APP_CFG.offline_db_path, word + ext)
            if os.path.exists(p):
                self.batch_query_path = p
                self.lbl_batch_query.setText(os.path.basename(p))
                self.lbl_batch_query.setStyleSheet("color: #10b981; font-weight: bold;")
                self.log_status(f"✅ Vzorek '{word}' načten z databáze.")
                return
                
        QMessageBox.warning(self, "Chyba", f"Slovo '{word}' nebylo v lokální databázi nalezeno.\n(Cesta: {APP_CFG.offline_db_path})")

    def select_batch_query(self):
        path, _ = QFileDialog.getOpenFileName(self, "Vyberte vzorek hledaného slova", "", "Audio (*.mp3 *.wav *.m4a *.flac)")
        if path:
            self.batch_query_path = path
            name = os.path.basename(path)
            self.lbl_batch_query.setText(name)
            self.lbl_batch_query.setStyleSheet("color: #10b981; font-weight: bold;")
            
            clean_word = os.path.splitext(name)[0]
            self.input_batch_query.setText(clean_word)

    def log_batch_append(self, text):
        self.log_batch.append(text)
        sb = self.log_batch.verticalScrollBar()
        sb.setValue(sb.maximum())

    def run_batch_benchmark(self):
        if not hasattr(self, 'batch_long_path') or not self.batch_long_path:
            QMessageBox.warning(self, "Chyba", "Nejprve vyberte prohledávané dlouhé audio.")
            return
        if not hasattr(self, 'batch_query_path') or not self.batch_query_path:
            QMessageBox.warning(self, "Chyba", "Vyberte konkrétní audio vzorek slova.")
            return
            
        # Zjištění a validace Ground Truth z nového ComboBoxu
        gt_text = self.combo_gt.currentText().strip()
        if os.path.exists(gt_text):
            self.ground_truth_path = gt_text
        else:
            base_dir = os.path.dirname(APP_CFG.offline_db_path)
            possible_path = os.path.join(base_dir, gt_text)
            if os.path.exists(possible_path):
                self.ground_truth_path = possible_path
            else:
                QMessageBox.warning(self, "Chybí Standard", "Vybraný zlatý standard na disku neexistuje.\nUjistěte se, že je vygenerovaný ve složce s daty, nebo jej vyberte ručně přes 'Procházet...'.")
                return
                
        self.txt_batch_detail.clear()
        self.log_batch.clear()
        self.btn_run_batch.setEnabled(False)
        
        method = self.combo_batch_method.currentData()
        thr = float(self.input_threshold.text())
        
        self.worker_batch = BatchEvaluationWorker(self.batch_long_path, self.batch_query_path, self.ground_truth_path, method, thr)
        self.worker_batch.progress.connect(lambda v, m: (self.progress_batch.setValue(v), self.lbl_batch_status.setText(m)))
        self.worker_batch.log_msg.connect(self.log_batch_append)
        self.worker_batch.result_row.connect(self.on_batch_row)
        self.worker_batch.finished.connect(lambda: self.btn_run_batch.setEnabled(True))
        self.worker_batch.start()

    def on_batch_row(self, data):
        self.batch_detailed_data[data['word']] = data
        row = self.table_batch.rowCount()
        self.table_batch.insertRow(row)
        
        cell_widget = QWidget()
        cell_layout = QHBoxLayout(cell_widget)
        cell_layout.setContentsMargins(5, 2, 5, 2)
        cell_layout.setSpacing(10)
        
        lbl_word = QLabel(data['word'])
        lbl_word.setStyleSheet("font-weight: bold;")
        
        btn_play = QPushButton()
        btn_play.setToolTip("Přehrát nálezy")
        btn_play.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_play.setFixedSize(30, 24)
        
        icon_path = os.path.join("assets", "audio.png") 
        if os.path.exists(icon_path):
            btn_play.setIcon(QIcon(icon_path))
            btn_play.setIconSize(QSize(18, 18)) 
        else:
            btn_play.setText("🔊")

        btn_play.setStyleSheet("""
            QPushButton { 
                background-color: transparent; 
                border: none; 
            }
            QPushButton:hover { 
                background-color: rgba(16, 185, 129, 0.2); 
                border-radius: 4px;
            }
        """)
        
        btn_play.clicked.connect(lambda checked, w=data['word']: self.play_exhaustive_word(w))
        
        cell_layout.addWidget(lbl_word)
        cell_layout.addStretch()
        cell_layout.addWidget(btn_play)
        
        self.table_batch.setCellWidget(row, 0, cell_widget)
        
        self.table_batch.setItem(row, 1, QTableWidgetItem(str(data.get('gt_count', 0))))
        self.table_batch.setItem(row, 2, QTableWidgetItem(str(data.get('found_count', 0))))
        self.table_batch.setItem(row, 3, QTableWidgetItem(f"{data.get('frr', 0) * 100:.1f}%"))
        self.table_batch.setItem(row, 4, QTableWidgetItem(f"{data.get('fa_h', 0):.2f}"))
        
        # Výpočet F1 skóre
        tp = data.get('tp', 0)
        fp = data.get('fp', 0)
        fn = data.get('gt_count', 0) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        self.table_batch.setItem(row, 5, QTableWidgetItem(f"{f1:.2f}"))

    def on_batch_row_selected(self):
        items = self.table_batch.selectedItems()
        if not items: return
        
        row_idx = items[0].row()
        word = None
        
        widget = self.table_batch.cellWidget(row_idx, 0)
        if widget:
            word = widget.findChild(QLabel).text()
        
        if not word: return
        
        data = self.batch_detailed_data.get(word)
        if not data: return

        gt_times_str = ", ".join([f"{t}s" for t in data.get('gt_times', [])]) if data.get('gt_times') else "Žádný výskyt"
        app_times_str = ", ".join([f"{t}s" for t in data.get('found_times', [])]) if data.get('found_times') else "Nic nenalezeno"

        html = f"""
        <h3 style="color: #10b981; margin-bottom: 5px;">Slovo: '{word.upper()}'</h3>
        <p><b>Skutečné časy v nahrávce dle zlatého standardu:</b><br>
        <span style="color: #4b5563;">{gt_times_str}</span></p>
        
        <p><b>Časy nalezené aplikací:</b><br>
        <span style="color: #4b5563;">{app_times_str}</span></p>
        """
        self.txt_batch_detail.setHtml(html)

    def play_exhaustive_word(self, word):
        data = self.batch_detailed_data.get(word)
        if not data: return
            
        times = data.get("found_times", [])
        if not times:
            self.log_status(f"Pro slovo '{word}' nejsou k dispozici žádné časy.")
            return

        timestamp_s = None

        if len(times) == 1:
            timestamp_s = times[0]
            self.log_status(f"Nalezen 1 výskyt slova '{word}', automaticky přehrávám čas {timestamp_s}s.")
        else:
            items = [f"Čas: {t} s" for t in times]
            item, ok = QInputDialog.getItem(self, f"Přehrát '{word}'", f"Nalezeno {len(times)} výskytů. Vyberte čas:", items, 0, False)
            if ok and item:
                idx = items.index(item)
                timestamp_s = times[idx]

        if timestamp_s is not None:
            try:
                y, sr = librosa.load(self.batch_long_path, sr=16000)
                start_sample = int(timestamp_s * sr)
                end_sample = start_sample + int(1.0 * sr) 
                pad = int(0.05 * sr)
                y_cut = y[max(0, start_sample-pad) : min(len(y), end_sample+pad)]
                self.play_mono(y_cut, sr, f"Kvantitativní nález: {word} ({timestamp_s}s)", "E")
            except Exception as e:
                self.log_status(f"❌ Chyba při přehrávání: {e}")

    def select_file(self, type_):
        path, _ = QFileDialog.getOpenFileName(self, "Vyber audio", "", "Audio (*.mp3 *.wav *.m4a *.flac)")
        if not path: return
        name = os.path.basename(path)
        
        if type_ == 'long_single': 
            self.path_long_single = path
            self.input_single_long.setText(name)
            self.lbl_single_long.setText(name)
            self.lbl_single_long.setStyleSheet("color: #10b981; font-weight: bold; font-size: 9pt;")
        elif type_ == 'query_single': 
            self.path_query = path
            self.input_query.setText(name)
            self.lbl_single_query.setText(name)
            self.lbl_single_query.setStyleSheet("color: #10b981; font-weight: bold; font-size: 9pt;")
        elif type_ == 'long_corpus': 
            self.path_long_corpus = path
            self.lbl_corpus_long.setText(name)
            # NOVÉ: Načte text, když uživatel vybere soubor pro korpus
            self.display_ground_truth_text(path)

    def load_single_long_from_folder(self):
        audio_name = self.input_single_long.text().strip()
        if not audio_name: return
        if audio_name.isdigit():
            audio_name = f"sample-{int(audio_name):06d}"
            self.input_single_long.setText(audio_name)
            
        base_dir = os.path.dirname(APP_CFG.offline_db_path)
        possible_dirs = [APP_CFG.offline_db_path, base_dir, os.path.join(base_dir, "cv_16k_wav"), 
                         os.path.join(base_dir, "Common Voice Spontaneous Speech 3.0 - English")]
        
        for d in possible_dirs:
            for ext in APP_CFG.supported_extensions:
                p = os.path.join(d, audio_name + ext)
                if os.path.exists(p):
                    self.path_long_single = p
                    self.lbl_single_long.setText(os.path.basename(p))
                    self.lbl_single_long.setStyleSheet("color: #10b981; font-weight: bold; font-size: 9pt;")
                    self.log_status(f"✅ Audio '{audio_name}' načteno.")
                    return
        QMessageBox.warning(self, "Chyba", f"Audio '{audio_name}' nebylo nalezeno.")

    def load_query_from_folder(self):
        word = self.input_query.text().strip()
        if not word: return
        for ext in APP_CFG.supported_extensions:
            p = os.path.join(APP_CFG.offline_db_path, word + ext)
            if os.path.exists(p):
                self.path_query = p
                self.lbl_single_query.setText(os.path.basename(p))
                self.lbl_single_query.setStyleSheet("color: #10b981; font-weight: bold; font-size: 9pt;")
                self.log_status(f"✅ Vzorek '{word}' dohledán.")
                return
        QMessageBox.warning(self, "Chyba", f"Slovo '{word}' nebylo nalezeno.")

    def run_single_analysis(self):
        self.log_text.clear()
        self.single_found_ranges = [] 
        self.btn_analyze.setEnabled(False)
        
        # Zablokování všech interakčních tlačítek během výpočtu
        self.btn_next.setEnabled(False)
        self.btn_visual.setEnabled(False)
        for btn in [self.btn_listen_L, self.btn_listen_stereo, self.btn_listen_R]:
            btn.setEnabled(False)
            
        self.progress_detail.setValue(0)
        
        if not hasattr(self, 'path_long_single') or not hasattr(self, 'path_query'):
            self.log_text.append("❌ Chyba: Chybí cesta k audiu nebo vzorku.")
            self.btn_analyze.setEnabled(True)
            return

        try:
            method = self.combo_method.currentData()
            self.log_status(f"Startuji proces analýzy (Metoda: {method})...")
            self.log_status("Načítám audio korpusu z disku a provádím ořez ticha (VAD)...")
            y_long, sr_long = load_and_prep(self.path_long_single)
            self.log_status("Načítám hledaný vzorek a provádím ořez ticha (VAD)...")
            y_query, sr_query = load_and_prep(self.path_query)
            
            from core.audio_utils import get_wav2vec_features, get_mfcc_features, get_math_spectrogram
            
            self.log_status(f"Extrahuji akustické rysy ze vzorku...")
            if method == "wav2vec":
                query_features = get_wav2vec_features(y_query, sr_query)
                self.log_status(f"Extrahuji Wav2Vec rysy z celé nahrávky (Délka: {len(y_long)/sr_long:.1f} s)...")
                long_features = get_wav2vec_features(y_long, sr_long)
            elif method == "dtw":
                query_features = get_mfcc_features(y_query, sr_query)
                self.log_status(f"Extrahuji MFCC rysy z celé nahrávky...")
                long_features = get_mfcc_features(y_long, sr_long)
            else:
                self.log_status("Tento model používá asynchronní Worker, sledujte logy...")
                self.start_single_worker()
                return

            self.log_status("Spouštím prohledávací algoritmus (DTW). Tohle může chvíli trvat...")
            self.start_single_worker()

        except Exception as e:
            self.log_text.append(f"❌ Došlo k chybě během analýzy: {str(e)}")
            self.btn_analyze.setEnabled(True)

    def run_single_next(self): 
        # HLEDÁNÍ DALŠÍHO VÝSKYTU -> Nemazat single_found_ranges!
        self.log_status("Pokračuji v hledání dalšího výskytu...")
        self.start_single_worker()

    def start_single_worker(self):
        self.btn_analyze.setEnabled(False)
        self.btn_next.setEnabled(False) # Deaktivace po dobu běhu
        
        # Zde předáváme self.single_found_ranges do Workeru (jako parametr excluded)
        self.worker1 = SingleAnalysisWorker(self.path_long_single, self.path_query, self.combo_method.currentData(), self.single_found_ranges)
        self.worker1.progress.connect(self.progress_detail.setValue)
        self.worker1.finished.connect(self.on_single_finished)
        self.worker1.start()

    def on_single_finished(self, res: MatchResult):
        self.btn_analyze.setEnabled(True)
        self.progress_detail.setValue(100)
        
        if not res.is_success:
            if "(již) nalezeno" in (res.error or ""):
                self.log_text.append("Hledání dokončeno.")
                self.show_msg("Konec hledání", "(již) nalezeno", QMessageBox.Icon.Information)
            else:
                # Vypíše celý dlouhý strom chyby do konzole v aplikaci
                self.log_text.append(f"\n❌ DETAILNÍ VÝPIS CHYBY:\n{res.error}\n")
                
                # V popupu ukáže jen varování
                self.show_msg("Kritická chyba", "Aplikace spadla. Podívejte se do konzole dole na celý výpis chyby.", QMessageBox.Icon.Critical)
            return
            
        self.log_text.append(f"✅ Nalezeno! Čas: {res.start_f / (res.sr / AUDIO_CFG.hop_length):.2f} s")
        self.single_current_result = res
        self.single_found_ranges.append((res.start_f, res.end_f))
        
        # Odemknutí všech tlačítek, protože máme platný výsledek
        self.btn_next.setEnabled(True)
        self.btn_visual.setEnabled(True)
        for btn in [self.btn_listen_L, self.btn_listen_stereo, self.btn_listen_R]:
            btn.setEnabled(True)
        
        try:
            self.draw_results(res, self.canvas_single, self.single_found_ranges)
        except Exception as e:
            self.log_text.append(f"❌ Chyba při vykreslování: {e}")

    def update_corpus_progress(self, val, msg):
        self.progress_corpus.setValue(val)
        if msg:
            self.log_corpus.append(f"{msg}")
            sb = self.log_corpus.verticalScrollBar()
            sb.setValue(sb.maximum())

    def run_corpus_scan(self):
        if not hasattr(self, 'path_long_corpus') or not self.path_long_corpus:
            QMessageBox.warning(self, "Chyba", "Nejprve vyberte dlouhé audio.")
            return
            
        self.btn_corpus_scan.setEnabled(False)
        self.btn_corpus_pause.setEnabled(True)
        self.log_corpus.clear()
        self.progress_corpus.setValue(0)
        self.table.setRowCount(0)
        
        self.worker2 = CorpusScannerWorker(self.path_long_corpus, APP_CFG.database_file, self.combo_corpus_method.currentData())
        self.worker2.progress.connect(self.update_corpus_progress)
        self.worker2.finished.connect(self.on_corpus_scan_finished)
        self.worker2.start()

    def on_corpus_scan_finished(self, res):
        self.btn_corpus_scan.setEnabled(True)
        self.btn_corpus_pause.setEnabled(False)
        self.progress_corpus.setValue(100)
        
        if res["type"] == "error": 
            self.log_corpus.append(f"❌ Chyba: {res.get('msg', 'Neznámá chyba')}")
            return

        whisper_text = res.get('whisper_text', '')
        if whisper_text:
            self.lbl_whisper_text.setText(f"<b>Surový přepis Whisperu:</b> {whisper_text}")
            self.lbl_whisper_text.show()
            self.frame_corpus_gt.show()
           
        self.log_corpus.append("✅ Analýza úspěšně dokončena! Vykresluji výsledky...")
        
        all_results = res['results']
        
        MAX_SAFE = 1500
        if len(all_results) > MAX_SAFE:
            self.log_corpus.append(f"⚠ Pozor: Nalezeno {len(all_results)} shod! Vykresluji {MAX_SAFE} nejlepších, aby aplikace nespadla.")
            is_pattern = self.combo_corpus_method.currentData() == 'pattern'
            all_results = sorted(all_results, key=lambda x: x['score'], reverse=is_pattern)[:MAX_SAFE]

        final_results = sorted(all_results, key=lambda x: x['start_f'])
        
        self.corpus_data = res
        self.corpus_data['results'] = final_results 
        
        self.table.setRowCount(len(final_results))
        
        for i, row in enumerate(final_results):
            self.corpus_history[row['word']] = [(row['start_f'], row['end_f'])]
            
            cell_widget = QWidget()
            cell_layout = QHBoxLayout(cell_widget)
            cell_layout.setContentsMargins(5, 2, 5, 2)
            cell_layout.setSpacing(10)
            
            lbl_word = QLabel(row['word'])
            lbl_word.setStyleSheet("font-weight: bold;")
            
            btn_play = QPushButton()
            btn_play.setToolTip("Přehrát tento nález")
            btn_play.setCursor(Qt.CursorShape.PointingHandCursor)
            btn_play.setFixedSize(30, 24)
            
            icon_path = os.path.join("assets", "audio.png") 
            if os.path.exists(icon_path):
                btn_play.setIcon(QIcon(icon_path))
                btn_play.setIconSize(QSize(18, 18)) 
            else:
                btn_play.setText("🔊")

            btn_play.setStyleSheet("""
                QPushButton { background-color: transparent; border: none; }
                QPushButton:hover { background-color: rgba(16, 185, 129, 0.2); border-radius: 4px; }
            """)
            
            btn_play.clicked.connect(lambda checked, idx=i: self.play_specific_corpus_word(idx))
            
            cell_layout.addWidget(lbl_word)
            cell_layout.addStretch()
            cell_layout.addWidget(btn_play)
            
            self.table.setCellWidget(i, 0, cell_widget)
            self.table.setItem(i, 1, QTableWidgetItem(f"{row['score']:.4f}"))
            
            time_sec = row['start_f'] * AUDIO_CFG.hop_length / res['sr']
            self.table.setItem(i, 2, QTableWidgetItem(f"{time_sec:.1f}"))

    def on_corpus_method_changed(self):
        """Schová přepis Whisperu, pokud se změní model nebo je vybrán ne-Whisper model."""
        self.lbl_whisper_text.hide()
        self.lbl_whisper_text.setText("")
        # Pokud v GT rámci nic nezbylo (není tam ani zlatý standard), schováme celý rámec
        if not self.lbl_corpus_gt.text():
            self.frame_corpus_gt.hide()

    def toggle_corpus_pause(self):
        """Přepíná stav pauzy ve workeru a mění text tlačítka."""
        if hasattr(self, 'worker2') and self.worker2.isRunning():
            is_paused = self.worker2.toggle_pause()
            self.btn_corpus_pause.setText("POKRAČOVAT" if is_paused else "PAUZA")
            # Vizuální odlišení pauzy
            if is_paused:
                self.btn_corpus_pause.setStyleSheet("background-color: #fef3c7; color: #92400e; font-weight: bold;")
            else:
                self.btn_corpus_pause.setStyleSheet("")

    def play_specific_corpus_word(self, row_idx):
        if not self.corpus_data or 'results' not in self.corpus_data: return

        row_data = self.corpus_data['results'][row_idx]
        word = row_data['word']
        sr = self.corpus_data.get('sr', 16000)

        start_sample = int(row_data['start_f'] * AUDIO_CFG.hop_length)
        end_sample = int(row_data['end_f'] * AUDIO_CFG.hop_length)

        self.log_corpus.append(f"Přehrávám '{word}': snímky {start_sample} až {end_sample}")

        padding_samples = int(0.05 * sr) 
        start_sample = max(0, start_sample - padding_samples)

        if 'y_long' in self.corpus_data:
            y_audio = self.corpus_data['y_long']
        else:
            self.log_corpus.append("Načítám audio do paměti pro přehrávání...")
            QApplication.processEvents()
            
            y_audio, sr = librosa.load(self.path_long_corpus, sr=sr)
            self.corpus_data['y_long'] = y_audio 

        end_sample = min(len(y_audio), end_sample + padding_samples)
        y_match = y_audio[start_sample:end_sample]

        self.play_mono(y_match, sr, f"Slovo z korpusu: '{word}'", "C")

    def draw_results(self, res: MatchResult, canvas, found_ranges):
        canvas.figure.clear()
        ax1, ax2 = canvas.figure.subplots(1, 2)

        ax1.set_facecolor('#000000')
        ax2.set_facecolor('#000000')
        
        frames_per_sec = res.sr / AUDIO_CFG.hop_length
        target_frames = int(10.0 * frames_per_sec)
        
        def pad_with_min(array, pad_l, pad_r, min_val):
            if pad_l <= 0 and pad_r <= 0:
                return array
            return np.pad(array, ((0, 0), (max(0, pad_l), max(0, pad_r))), mode='constant', constant_values=min_val)

        num_mels = res.vis_spec_query.shape[0]
        y_ticks = np.arange(0, num_mels + 1, 16)
        x_ticks_pos = np.arange(0, 11, 1)

        # ---------------------------------------------
        # LEVÝ GRAF (Hledaný vzorek)
        # ---------------------------------------------
        query_frames = res.vis_spec_query.shape[1]
        min_val_q = np.min(res.vis_spec_query)
        
        if query_frames < target_frames:
            pad_total = target_frames - query_frames
            pad_left_q = pad_total // 2
            pad_right_q = pad_total - pad_left_q
            spec_query_10s = pad_with_min(res.vis_spec_query, pad_left_q, pad_right_q, min_val_q)
        else:
            start_idx = (query_frames - target_frames) // 2
            spec_query_10s = res.vis_spec_query[:, start_idx : start_idx + target_frames]
            
        img1 = librosa.display.specshow(
            spec_query_10s, ax=ax1, x_axis='s', sr=res.sr, hop_length=AUDIO_CFG.hop_length
        )
        ax1.set_title("Hledaný vzorek")
        ax1.set_xlabel("Čas vzorku (s)")
        
        # Popisek osy Y a ticks POUZE u levého grafu
        ax1.set_ylabel("Mel pásmo (Index)")
        ax1.set_yticks(y_ticks)
        ax1.tick_params(axis='y', labelleft=True)
        
        ax1.set_ylim(0, num_mels)
        ax1.set_xticks(x_ticks_pos)
        ax1.set_xlim(0, 10.0)
        ax1.margins(x=0, y=0)
        
        # ---------------------------------------------
        # PRAVÝ GRAF (Detail nálezu z Korpusu)
        # ---------------------------------------------
        start_s = res.start_f / frames_per_sec
        page_start_s = float(int(start_s // 10) * 10)
        page_start_frames = int(page_start_s * frames_per_sec)
        page_end_frames = page_start_frames + target_frames
        
        max_frames = res.vis_spec_long.shape[1]
        min_val_long = np.min(res.vis_spec_long)
        
        valid_start = max(0, page_start_frames)
        valid_end = min(max_frames, page_end_frames)
        
        valid_slice = res.vis_spec_long[:, valid_start:valid_end]
        pad_left_long = max(0, -page_start_frames)
        pad_right_long = max(0, page_end_frames - max_frames)
        
        spec_slice_10s = pad_with_min(valid_slice, pad_left_long, pad_right_long, min_val_long)
            
        img2 = librosa.display.specshow(
            spec_slice_10s, ax=ax2, x_axis='s', sr=res.sr, hop_length=AUDIO_CFG.hop_length
        )
        
        ax2.set_title(f"Detail nálezu | Skutečný čas v nahrávce: {start_s:.2f} s")
        ax2.set_xlabel("Absolutní čas nahrávky (s)")
        
        ax2.set_ylabel("")
        ax2.set_yticks(y_ticks)
        ax2.tick_params(axis='y', labelleft=True)
        
        ax2.set_ylim(0, num_mels)
        ax2.margins(x=0, y=0)
        
        x_ticks_labels = [str(int(page_start_s + i)) for i in x_ticks_pos]
        ax2.set_xticks(x_ticks_pos)
        ax2.set_xticklabels(x_ticks_labels)
        ax2.set_xlim(0, 10.0)
        
        for i, (f_start, f_end) in enumerate(found_ranges):
            s_start = f_start / frames_per_sec
            s_end = f_end / frames_per_sec
            
            if s_end > page_start_s and s_start < page_start_s + 10.0:
                local_start_s = max(0.0, s_start - page_start_s)
                local_end_s = min(10.0, s_end - page_start_s)
                
                # Zvýraznění aktuálního nálezu zeleně, starších šedě
                if i == len(found_ranges) - 1:
                    ax2.axvspan(local_start_s, local_end_s, color='#10b981', alpha=0.4)
                else:
                    ax2.axvspan(local_start_s, local_end_s, color='#9ca3af', alpha=0.5)

        # --- ZDE PŘIDÁVÁME LEGENDU K PRAVÉMU GRAFU (ax2) ---
        canvas.figure.colorbar(img2, ax=ax2, format="%+2.0f dB", fraction=0.046, pad=0.04)

        # Úprava okrajů, aby se legenda vpravo vešla
        canvas.figure.subplots_adjust(left=0.06, right=0.92, bottom=0.12, top=0.90, wspace=0.10) 
        canvas.draw()
    # ==========================================
    # 🔊 ZVUKOVÉ FUNKCE
    # ==========================================
    def create_listen_buttons(self, play_func):
        btn_L = QPushButton("L (Vzorek)"); btn_L.clicked.connect(lambda: play_func('left'))
        btn_S = QPushButton("Přehrát (Stereo)"); btn_S.clicked.connect(lambda: play_func('stereo'))
        btn_R = QPushButton("P (Nález)"); btn_R.clicked.connect(lambda: play_func('right'))
        return btn_L, btn_S, btn_R

    def play_stereo_match(self, y_query: np.ndarray, y_match: np.ndarray, sr: int):
        if y_query is None or y_match is None or len(y_query) == 0 or len(y_match) == 0:
            return

        target_sr = 48000
        y_q = librosa.resample(librosa.util.normalize(y_query), orig_sr=sr, target_sr=target_sr)
        y_m = librosa.resample(librosa.util.normalize(y_match), orig_sr=sr, target_sr=target_sr)

        silence = np.zeros(int(0.2 * target_sr), dtype=np.float32)
        y_q = np.concatenate((silence, y_q))
        y_m = np.concatenate((silence, y_m))

        max_len = max(len(y_q), len(y_m))
        left = np.pad(y_q, (0, max_len - len(y_q)))
        right = np.pad(y_m, (0, max_len - len(y_m)))
        stereo_signal = np.vstack([left, right]).T

        self._execute_windows_native_playback(stereo_signal, target_sr, "stereo")

    def play_mono(self, y_audio: np.ndarray, sr: int, log_msg: str, side: str):
        if y_audio is None or len(y_audio) == 0: return
        
        target_sr = 48000
        y_norm = librosa.resample(librosa.util.normalize(y_audio), orig_sr=sr, target_sr=target_sr)
        
        silence = np.zeros(int(0.2 * target_sr), dtype=np.float32)
        y_final = np.concatenate((silence, y_norm))
        
        stereo_signal = np.vstack((y_final, y_final)).T
        
        if hasattr(self, 'log_text') and self.log_text.isVisible():
            self.log_text.append(f"▶ Přehrávám: {log_msg}")

        self._execute_windows_native_playback(stereo_signal, target_sr, f"mono_{side}")

    def _execute_windows_native_playback(self, signal: np.ndarray, sr: int, label: str):
        try:
            tmp_file = os.path.join(tempfile.gettempdir(), f"shazam_native_{label}.wav")
            sf.write(tmp_file, signal, sr, subtype='PCM_16')
            winsound.PlaySound(None, winsound.SND_PURGE)
            winsound.PlaySound(tmp_file, winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception as e:
            if hasattr(self, 'log_text') and self.log_text.isVisible():
                self.log_text.append(f"❌ Chyba nativního přehrávání: {e}")
                
    def play_single(self, mode: str):
        if not hasattr(self, 'single_current_result') or not self.single_current_result:
            return

        res = self.single_current_result
        padding_samples = int(APP_CFG.playback_padding_sec * res.sr)
        start_sample = max(0, int(res.start_f * AUDIO_CFG.hop_length) - padding_samples)
        end_sample = min(len(res.y_long), int(res.end_f * AUDIO_CFG.hop_length) + padding_samples)
        y_match = res.y_long[start_sample:end_sample]

        if mode == 'stereo':
            self.play_stereo_match(res.y_query, y_match, res.sr)
        elif mode == 'left':
            self.play_mono(res.y_query, res.sr, "Hledaný vzorek (Levý graf)", "L")
        elif mode == 'right':
            self.play_mono(y_match, res.sr, "Detail nálezu (Pravý graf)", "P")

    def play_corpus(self, m): 
        pass
    
    def open_settings(self):
        dlg = SettingsDialog(self)
        dlg.exec()
        
    def open_visual_comparison(self, res):
        if not res: return
        dlg = VisualCompareDialog(res, self)
        dlg.exec()
        
    def apply_native_titlebar_color(self, target_widget=None, hex_bg="#10b981", hex_text="#ffffff"):
        if sys.platform != "win32":
            return
            
        # Pokud není specifikován widget, obarvíme hlavní okno (self)
        widget = target_widget if target_widget else self
        
        try:
            DWMWA_CAPTION_COLOR = 35
            DWMWA_TEXT_COLOR = 36
            hwnd = HWND(int(widget.winId()))
            
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

    def show_msg(self, title, text, icon=QMessageBox.Icon.Information):
        """Vytvoří QMessageBox, obarví mu lištu a zobrazí ho."""
        msg = QMessageBox(self)
        msg.setWindowTitle(title)
        msg.setText(text)
        msg.setIcon(icon)
        
        # Aplikujeme zelenou lištu předtím, než se dialog ukáže
        self.apply_native_titlebar_color(target_widget=msg)
        msg.exec()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudioMatcherApp()
    window.show()
    sys.exit(app.exec())