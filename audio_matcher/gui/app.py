# ==============================================================================
# audio_matcher/gui/app.py
# ==============================================================================
import os
import sys
import tempfile
import ctypes
import numpy as np
import soundfile as sf
import winsound
from ctypes.wintypes import HWND, DWORD

from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QComboBox,
                             QTextEdit, QMessageBox, QLineEdit, QTabWidget,
                             QTableWidget, QTableWidgetItem, QHeaderView,
                             QProgressBar, QSplitter, QFrame, QApplication,
                             QInputDialog)
from PyQt6.QtCore import Qt
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtGui import QIcon

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import librosa.display

from config import APP_CFG, AUDIO_CFG, MODERN_STYLESHEET, COMMON_STYLE_INPUTS, COMMON_STYLE_BUTTONS
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
        
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.audio_output.setVolume(1.0)
        self.player.setAudioOutput(self.audio_output)

        self.single_found_ranges = []
        self.single_current_result: MatchResult | None = None
        self.corpus_data = None
        self.corpus_history = {}
        self.batch_detailed_data = {} 
        
        self.init_ui()
        self.apply_native_titlebar_color()
        self.center_window()

    def log_status(self, message: str):
        self.log_text.append(f"{message}")
        QApplication.processEvents()
    
    def center_window(self):
        qr = self.frameGeometry()
        cp = self.screen().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
        
    def init_ui(self):
        main_widget = QWidget()
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(15, 15, 15, 15)
        
        if os.path.exists(APP_CFG.icon_path):
            self.setWindowIcon(QIcon(APP_CFG.icon_path))

        self.tabs = QTabWidget()
        self.btn_settings = QPushButton("Nastavení")
        self.btn_settings.setStyleSheet("QPushButton { background: transparent; border: none; font-weight: 500; color: #6b7280; padding: 10px 20px; } QPushButton:hover { color: #10b981; }")
        self.btn_settings.clicked.connect(self.open_settings)
        self.tabs.setCornerWidget(self.btn_settings, Qt.Corner.TopRightCorner)

        self.tab_detail, self.tab_corpus, self.tab_batch = QWidget(), QWidget(), QWidget()
        self.setup_detail_tab()
        self.setup_corpus_tab()
        self.setup_batch_tab()
        
        self.tabs.addTab(self.tab_detail, "Srovnávací analýza")
        self.tabs.addTab(self.tab_corpus, "Analýza korpusu")
        self.tabs.addTab(self.tab_batch, "Kvantitativní evaluace")
        
        layout.addWidget(self.tabs)
        self.setCentralWidget(main_widget)

    def setup_detail_tab(self):
        layout = QVBoxLayout(self.tab_detail)
        layout.setContentsMargins(20, 20, 20, 20)
        
        input_frame = QFrame(); input_frame.setObjectName("cardPanel")
        input_layout = QVBoxLayout(input_frame)
        
        title_detail = QLabel("Srovnávací analýza"); title_detail.setStyleSheet("font-size: 14pt; font-weight: bold;")
        desc_detail = QLabel("Algoritmus porovná hledaný vzorek s prohledávaným audiem a detailně vizualizuje shodu.")
        desc_detail.setStyleSheet("color: #6b7280; margin-bottom: 10px;")
        input_layout.addWidget(title_detail); input_layout.addWidget(desc_detail)

        def create_file_row(label_text, placeholder, browse_cb):
            row = QHBoxLayout()
            lbl = QLabel(label_text); lbl.setFixedWidth(130)
            inp = QLineEdit(); inp.setPlaceholderText(placeholder); inp.setFixedWidth(375)
            if os.path.exists(APP_CFG.search_icon_path):
                inp.addAction(QIcon(APP_CFG.search_icon_path), QLineEdit.ActionPosition.TrailingPosition).triggered.connect(inp.returnPressed)
            inp.setStyleSheet(COMMON_STYLE_INPUTS)
            btn = QPushButton("Procházet vlastní..."); btn.setStyleSheet(COMMON_STYLE_BUTTONS); btn.clicked.connect(browse_cb)
            status = QLabel("Nebylo vybráno"); status.setStyleSheet("color: #6b7280; font-style: italic;")
            row.addWidget(lbl); row.addWidget(inp); row.addWidget(btn); row.addWidget(status); row.addStretch()
            return row, inp, status

        row1, self.input_single_long, self.lbl_single_long = create_file_row("Prohledávané Audio:", "Zadejte číslo od 0 do 4075", lambda: self.select_file('long_single'))
        self.input_single_long.returnPressed.connect(self.load_single_long_from_folder)
        input_layout.addLayout(row1)

        row2, self.input_query, self.lbl_single_query = create_file_row("Hledaný Vzorek:", "Hledejte vzorek v DB", lambda: self.select_file('query_single'))
        self.input_query.returnPressed.connect(self.load_query_from_folder)
        input_layout.addLayout(row2)

        row3 = QHBoxLayout()
        lbl_model = QLabel("Analytický model:"); lbl_model.setFixedWidth(130)
        self.combo_method = QComboBox(); self.combo_method.setFixedWidth(375); self.combo_method.setStyleSheet(COMMON_STYLE_INPUTS)
        self.combo_method.addItems(["OpenAI Whisper (ASR)", "Wav2Vec 2.0 + DTW", "MFCC + DTW", "Pattern Matching"])
        self.combo_method.setItemData(0, "whisper"); self.combo_method.setItemData(1, "wav2vec"); self.combo_method.setItemData(2, "dtw"); self.combo_method.setItemData(3, "pattern")
        row3.addWidget(lbl_model); row3.addWidget(self.combo_method); row3.addStretch()
        input_layout.addLayout(row3)

        controls_layout = QHBoxLayout()
        self.btn_analyze = QPushButton("SPUSTIT ANALÝZU"); self.btn_analyze.setObjectName("primaryButton"); self.btn_analyze.clicked.connect(self.run_single_analysis)
        self.btn_single_pause = QPushButton("Pauza"); self.btn_single_pause.setStyleSheet(COMMON_STYLE_BUTTONS); self.btn_single_pause.setEnabled(False); self.btn_single_pause.clicked.connect(self.toggle_single_pause)
        self.btn_single_stop = QPushButton("Zrušit"); self.btn_single_stop.setStyleSheet("QPushButton { color: #dc2626; font-weight: bold; background: #f3f4f6; border: 1px solid #d1d5db; padding: 6px 15px;}"); self.btn_single_stop.setEnabled(False); self.btn_single_stop.clicked.connect(self.stop_single_scan)
        self.btn_next = QPushButton("Zobrazit další nález"); self.btn_next.setStyleSheet(COMMON_STYLE_BUTTONS); self.btn_next.setEnabled(False); self.btn_next.clicked.connect(self.run_single_next)
        self.btn_visual = QPushButton("Vizuální srovnání"); self.btn_visual.setStyleSheet(COMMON_STYLE_BUTTONS); self.btn_visual.setEnabled(False); self.btn_visual.clicked.connect(lambda: self.open_visual_comparison(self.single_current_result))
        
        controls_layout.addWidget(self.btn_analyze); controls_layout.addWidget(self.btn_single_pause); controls_layout.addWidget(self.btn_single_stop)
        controls_layout.addWidget(self.btn_next); controls_layout.addWidget(self.btn_visual); controls_layout.addStretch() 
        
        self.btn_listen_L, self.btn_listen_stereo, self.btn_listen_R = self.create_listen_buttons(self.play_single)
        for btn in [self.btn_listen_L, self.btn_listen_stereo, self.btn_listen_R]:
            btn.setStyleSheet(COMMON_STYLE_BUTTONS); btn.setEnabled(False); controls_layout.addWidget(btn)
            
        input_layout.addLayout(controls_layout)

        self.progress_detail = QProgressBar(); self.progress_detail.setFixedHeight(6); self.progress_detail.setTextVisible(False)
        self.log_text = QTextEdit(); self.log_text.setObjectName("logConsole"); self.log_text.setMaximumHeight(80)
        input_layout.addWidget(self.progress_detail); input_layout.addWidget(self.log_text)
        
        layout.addWidget(input_frame)
        self.canvas_single = FigureCanvas(Figure(figsize=(12, 5), dpi=100)); self.canvas_single.figure.patch.set_facecolor('#f3f4f6') 
        layout.addWidget(self.canvas_single, 1)

    def setup_corpus_tab(self):
        layout = QVBoxLayout(self.tab_corpus)
        layout.setContentsMargins(20, 20, 20, 20)
        
        input_frame = QFrame(); input_frame.setObjectName("cardPanel")
        input_layout = QVBoxLayout(input_frame)

        title_corpus = QLabel("Analýza korpusu"); title_corpus.setStyleSheet("font-size: 14pt; font-weight: bold;")
        desc_corpus = QLabel("Nástroj pro hromadné prohledání nahrávky a detekci všech výskytů podle zvoleného analytického modelu.")
        desc_corpus.setStyleSheet("color: #6b7280; margin-bottom: 10px;")
        input_layout.addWidget(title_corpus); input_layout.addWidget(desc_corpus)

        def create_file_row(label_text, placeholder, browse_cb):
            row = QHBoxLayout()
            lbl = QLabel(label_text); lbl.setFixedWidth(130)
            inp = QLineEdit(); inp.setPlaceholderText(placeholder); inp.setFixedWidth(375); inp.setStyleSheet(COMMON_STYLE_INPUTS)
            if os.path.exists(APP_CFG.search_icon_path):
                inp.addAction(QIcon(APP_CFG.search_icon_path), QLineEdit.ActionPosition.TrailingPosition).triggered.connect(inp.returnPressed)
            btn = QPushButton("Procházet vlastní..."); btn.setStyleSheet(COMMON_STYLE_BUTTONS); btn.clicked.connect(browse_cb)
            status = QLabel("Nebylo vybráno"); status.setStyleSheet("color: #6b7280; font-style: italic;")
            row.addWidget(lbl); row.addWidget(inp); row.addWidget(btn); row.addWidget(status); row.addStretch()
            return row, inp, status

        row1, self.input_corpus_long, self.lbl_corpus_long = create_file_row("Prohledávané audio:", "Zadejte číslo od 0 do 4075", lambda: self.select_file('long_corpus'))
        self.input_corpus_long.returnPressed.connect(self.load_corpus_long_from_folder)
        input_layout.addLayout(row1)

        row2 = QHBoxLayout()
        lbl_model = QLabel("Analytický model:"); lbl_model.setFixedWidth(130)
        self.combo_corpus_method = QComboBox(); self.combo_corpus_method.setFixedWidth(375); self.combo_corpus_method.setStyleSheet(COMMON_STYLE_INPUTS)
        self.combo_corpus_method.addItems(["OpenAI Whisper (ASR)", "Whisper + DTW Hybrid", "Wav2Vec 2.0 + DTW", "MFCC + DTW", "Pattern Matching"])
        self.combo_corpus_method.setItemData(0, "whisper"); self.combo_corpus_method.setItemData(1, "whisper_hybrid"); self.combo_corpus_method.setItemData(2, "wav2vec")
        self.combo_corpus_method.setItemData(3, "dtw"); self.combo_corpus_method.setItemData(4, "pattern")
        self.combo_corpus_method.currentIndexChanged.connect(self.on_corpus_method_changed)

        self.btn_corpus_pause = QPushButton("Pauza"); self.btn_corpus_pause.setEnabled(False); self.btn_corpus_pause.clicked.connect(self.toggle_corpus_pause)
        self.btn_corpus_stop = QPushButton("Zrušit"); self.btn_corpus_stop.setEnabled(False); self.btn_corpus_stop.setStyleSheet("color: #dc2626; font-weight: bold;"); self.btn_corpus_stop.clicked.connect(self.stop_corpus_scan)
        self.btn_corpus_scan = QPushButton("ANALYZOVAT KORPUS"); self.btn_corpus_scan.setObjectName("primaryButton"); self.btn_corpus_scan.clicked.connect(self.run_corpus_scan)

        row2.addWidget(lbl_model); row2.addWidget(self.combo_corpus_method); row2.addStretch(1)
        row2.addWidget(self.btn_corpus_pause); row2.addWidget(self.btn_corpus_stop); row2.addWidget(self.btn_corpus_scan)
        input_layout.addLayout(row2)

        self.progress_corpus = QProgressBar(); self.progress_corpus.setFixedHeight(6); self.progress_corpus.setTextVisible(False)
        self.log_corpus = QTextEdit(); self.log_corpus.setReadOnly(True); self.log_corpus.setObjectName("logConsole"); self.log_corpus.setMaximumHeight(80)

        self.frame_corpus_gt = QFrame(); self.frame_corpus_gt.setStyleSheet("QFrame { background-color: #ecfdf5; border: 1px solid #6ee7b7; border-radius: 6px; }")
        gt_layout = QVBoxLayout(self.frame_corpus_gt); gt_layout.setContentsMargins(10, 10, 10, 10)
        self.lbl_corpus_gt = QLabel(); self.lbl_corpus_gt.setWordWrap(True); self.lbl_corpus_gt.setStyleSheet("color: #065f46; font-size: 10pt;")
        self.lbl_whisper_text = QLabel(); self.lbl_whisper_text.setWordWrap(True); self.lbl_whisper_text.setStyleSheet("color: #065f46; font-size: 10pt;"); self.lbl_whisper_text.hide()
        gt_layout.addWidget(self.lbl_corpus_gt); gt_layout.addWidget(self.lbl_whisper_text)
        self.frame_corpus_gt.hide()

        self.table = QTableWidget(0, 3); self.table.setHorizontalHeaderLabels(["Slovo", "Skóre", "Čas (s)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch); self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)

        self.canvas_corpus = FigureCanvas(Figure(figsize=(8, 5), dpi=100)); self.canvas_corpus.figure.patch.set_facecolor('#f3f4f6') 

        layout.addWidget(input_frame); layout.addWidget(self.progress_corpus); layout.addWidget(self.log_corpus)
        layout.addWidget(self.frame_corpus_gt); layout.addWidget(self.table, 4); layout.addWidget(self.canvas_corpus, 1)

    def setup_batch_tab(self):
        layout = QVBoxLayout(self.tab_batch)
        layout.setContentsMargins(20, 20, 20, 20)

        batch_frame = QFrame(); batch_frame.setObjectName("cardPanel")
        batch_layout = QVBoxLayout(batch_frame)

        title_batch = QLabel("Kvantitativní evaluace"); title_batch.setStyleSheet("font-size: 14pt; font-weight: bold;")
        desc_batch = QLabel("Algoritmus projde nahrávku, najde všechny výskyty zadaného slova a vygeneruje statistiku vůči zlatému standardu.")
        desc_batch.setStyleSheet("color: #6b7280; margin-bottom: 10px;")
        batch_layout.addWidget(title_batch); batch_layout.addWidget(desc_batch)

        batch_controls = QVBoxLayout()
        def create_file_row(label_text, placeholder, browse_cb):
            row = QHBoxLayout()
            lbl = QLabel(label_text); lbl.setFixedWidth(130)
            if placeholder:
                inp = QLineEdit(); inp.setPlaceholderText(placeholder); inp.setFixedWidth(375); inp.setStyleSheet(COMMON_STYLE_INPUTS)
                if os.path.exists(APP_CFG.search_icon_path):
                    inp.addAction(QIcon(APP_CFG.search_icon_path), QLineEdit.ActionPosition.TrailingPosition).triggered.connect(inp.returnPressed)
            else:
                inp = QComboBox(); inp.setEditable(True); inp.setFixedWidth(375); inp.setStyleSheet(COMMON_STYLE_INPUTS)
            btn = QPushButton("Procházet vlastní..."); btn.setStyleSheet(COMMON_STYLE_BUTTONS); btn.clicked.connect(browse_cb)
            status = QLabel("Nebylo vybráno"); status.setStyleSheet("color: #6b7280; font-style: italic;")
            row.addWidget(lbl); row.addWidget(inp); row.addWidget(btn)
            if placeholder: row.addWidget(status)
            row.addStretch()
            return row, inp, status if placeholder else None

        row0, self.combo_gt, _ = create_file_row("Zlatý standard:", None, self.select_gt_file)
        self.combo_gt.addItems(["virtual_stream_ground_truth_complete_time.json", "virtual_stream_ground_truth.json"])
        batch_controls.addLayout(row0)

        row1, self.input_batch_long, self.lbl_batch_long = create_file_row("Prohledávané audio:", "Zadejte číslo od 0 do 4075", self.select_batch_long)
        self.input_batch_long.returnPressed.connect(self.load_batch_long_from_folder)
        batch_controls.addLayout(row1)

        row2, self.input_batch_query, self.lbl_batch_query = create_file_row("Hledaný vzorek:", "Hledejte vzorek v DB", self.select_batch_query)
        self.input_batch_query.returnPressed.connect(self.load_batch_query_from_folder)
        batch_controls.addLayout(row2)

        row3 = QHBoxLayout()
        lbl_model = QLabel("Analytický model:"); lbl_model.setFixedWidth(130)
        self.combo_batch_method = QComboBox(); self.combo_batch_method.setFixedWidth(375); self.combo_batch_method.setStyleSheet(COMMON_STYLE_INPUTS)
        self.combo_batch_method.addItems(["OpenAI Whisper (ASR)", "Wav2Vec 2.0 + DTW", "MFCC + DTW", "Pattern Matching"])
        self.combo_batch_method.setItemData(0, "whisper"); self.combo_batch_method.setItemData(1, "wav2vec"); self.combo_batch_method.setItemData(2, "dtw"); self.combo_batch_method.setItemData(3, "pattern")
        row3.addWidget(lbl_model); row3.addWidget(self.combo_batch_method); row3.addStretch()
        batch_controls.addLayout(row3)

        run_row = QHBoxLayout()
        self.btn_run_batch = QPushButton("SPUSTIT KVANTITATIVNÍ EVALUACI"); self.btn_run_batch.setObjectName("primaryButton"); self.btn_run_batch.clicked.connect(self.run_batch_benchmark)
        self.btn_batch_pause = QPushButton("Pauza"); self.btn_batch_pause.setStyleSheet(COMMON_STYLE_BUTTONS); self.btn_batch_pause.setEnabled(False); self.btn_batch_pause.clicked.connect(self.toggle_batch_pause)
        self.btn_batch_stop = QPushButton("Zrušit"); self.btn_batch_stop.setStyleSheet("QPushButton { color: #dc2626; font-weight: bold; background: #f3f4f6; border: 1px solid #d1d5db;}"); self.btn_batch_stop.setEnabled(False); self.btn_batch_stop.clicked.connect(self.stop_batch_scan)
        
        lbl_threshold = QLabel("Maximální práh:")
        self.input_threshold = QLineEdit("1.0"); self.input_threshold.setFixedWidth(50); self.input_threshold.setAlignment(Qt.AlignmentFlag.AlignCenter); self.input_threshold.setStyleSheet(COMMON_STYLE_INPUTS)
        
        run_row.addWidget(self.btn_run_batch); run_row.addWidget(self.btn_batch_pause); run_row.addWidget(self.btn_batch_stop)
        run_row.addStretch(); run_row.addWidget(lbl_threshold); run_row.addWidget(self.input_threshold)

        self.progress_batch = QProgressBar(); self.progress_batch.setFixedHeight(6); self.progress_batch.setTextVisible(False)
        self.lbl_batch_status = QLabel("Připraveno.")
        self.log_batch = QTextEdit(); self.log_batch.setReadOnly(True); self.log_batch.setObjectName("logConsole"); self.log_batch.setMaximumHeight(120)

        self.table_batch = QTableWidget(0, 11) 
        self.table_batch.setHorizontalHeaderLabels(["Slovo", "GT", "Nález", "TP", "FP", "FN", "Prec (%)", "Rec (%)", "FRR (%)", "FA/h", "F1"])
        self.table_batch.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for i in range(1, 11): self.table_batch.setColumnWidth(i, 60)
        self.table_batch.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table_batch.itemSelectionChanged.connect(self.on_batch_row_selected) 
        self.table_batch.setStyleSheet("QTableWidget { background-color: #ffffff; border: none; } QHeaderView::section:vertical { background-color: #f9fafb; color: #6b7280; border: none; border-right: 1px solid #e5e7eb; border-bottom: 1px solid #e5e7eb; padding: 0 8px; font-weight: bold; }")

        self.txt_batch_detail = QTextEdit(); self.txt_batch_detail.setReadOnly(True); self.txt_batch_detail.setObjectName("logConsole")

        batch_splitter = QSplitter(Qt.Orientation.Horizontal)
        batch_splitter.addWidget(self.table_batch); batch_splitter.addWidget(self.txt_batch_detail)
        batch_splitter.setSizes([600, 300]) 
        batch_splitter.setStyleSheet("QSplitter::handle { background-color: #10b981; width: 2px; }")

        batch_layout.addLayout(batch_controls); batch_layout.addLayout(run_row) 
        batch_layout.addWidget(self.progress_batch); batch_layout.addWidget(self.lbl_batch_status); batch_layout.addWidget(self.log_batch)
        batch_layout.addWidget(batch_splitter, 1)

        layout.addWidget(batch_frame, 1)

    def display_ground_truth_text(self, audio_path):
        import json
        filename = os.path.basename(audio_path)
        filename_no_ext = os.path.splitext(filename)[0]
        
        base_dir = os.path.dirname(APP_CFG.offline_db_path)
        gt_data, loaded_path = None, ""
        
        for d in [base_dir, os.path.dirname(base_dir), os.getcwd()]:
            for f_name in ["virtual_stream_ground_truth.json"]:
                p = os.path.join(d, f_name)
                if os.path.exists(p):
                    try:
                        with open(p, 'r', encoding='utf-8') as f:
                            gt_data = json.load(f)
                            loaded_path = p
                        break
                    except: pass
            if gt_data: break
                
        if not gt_data:
            self.log_corpus.append("ℹ️ Poznámka: Nepodařilo se najít JSON soubor se zlatým standardem.")
            self.frame_corpus_gt.hide(); return
            
        matching_words = []
        for w in gt_data:
            if not isinstance(w, dict): continue
            src = w.get('source_file') or w.get('filename') or ""
            if filename in src or filename_no_ext in src:
                text = w.get('orthographic') or w.get('word') or w.get('text')
                start_time = w.get('start') or w.get('stream_start') or 0.0
                if text: matching_words.append({"text": str(text), "start": float(start_time)})
                    
        if matching_words:
            matching_words.sort(key=lambda x: x["start"])
            self.lbl_corpus_gt.setText(f"<b>Přepis nahrávky '{filename}':</b> " + " ".join([item["text"].strip() for item in matching_words]))
            self.frame_corpus_gt.show()
            self.log_corpus.append(f"✅ Načten referenční text ze souboru: {os.path.basename(loaded_path)}")
        else:
            self.frame_corpus_gt.hide()
    
    def load_corpus_long_from_folder(self):
        audio_name = self.input_corpus_long.text().strip()
        if not audio_name: return
        if audio_name.isdigit():
            audio_name = f"sample-{int(audio_name):06d}"
            self.input_corpus_long.setText(audio_name)
            
        base_dir = os.path.dirname(APP_CFG.offline_db_path)
        for d in [APP_CFG.offline_db_path, base_dir, os.path.join(base_dir, "cv_16k_wav"), os.path.join(base_dir, "Common Voice Spontaneous Speech 3.0 - English")]:
            for ext in APP_CFG.supported_extensions:
                p = os.path.join(d, audio_name + ext)
                if os.path.exists(p):
                    self.path_long_corpus = p
                    self.lbl_corpus_long.setText(os.path.basename(p)); self.lbl_corpus_long.setStyleSheet("color: #10b981; font-weight: bold;")
                    self.log_status(f"✅ Audio korpusu načteno.")
                    self.display_ground_truth_text(p)
                    return
        QMessageBox.warning(self, "Chyba", f"Audio '{audio_name}' nebylo nalezeno.")
    
    def select_gt_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Vyberte zlatý standard", "", "JSON (*.json)")
        if path:
            self.combo_gt.setCurrentText(path)
            self.log_status(f"✅ Zlatý standard ručně vybrán.")

    def select_batch_long(self):
        path, _ = QFileDialog.getOpenFileName(self, "Vyberte dlouhé audio pro test", "", "Audio (*.mp3 *.wav *.m4a *.flac)")
        if path:
            self.batch_long_path = path
            self.lbl_batch_long.setText(os.path.basename(path)); self.lbl_batch_long.setStyleSheet("color: #10b981; font-weight: bold;")
            self.input_batch_long.setText(os.path.splitext(os.path.basename(path))[0])

    def load_batch_long_from_folder(self):
        audio_name = self.input_batch_long.text().strip()
        if not audio_name: return
        if audio_name.isdigit():
            audio_name = f"sample-{int(audio_name):06d}"
            self.input_batch_long.setText(audio_name) 
            
        base_dir = os.path.dirname(APP_CFG.offline_db_path)
        for d in [APP_CFG.offline_db_path, base_dir, os.path.join(base_dir, "cv_16k_wav"), os.path.join(base_dir, "Common Voice Spontaneous Speech 3.0 - English")]:
            for ext in APP_CFG.supported_extensions:
                p = os.path.join(d, audio_name + ext)
                if os.path.exists(p):
                    self.batch_long_path = p
                    self.lbl_batch_long.setText(os.path.basename(p)); self.lbl_batch_long.setStyleSheet("color: #10b981; font-weight: bold;")
                    self.log_status(f"✅ Audio korpusu načteno z DB.")
                    return
        QMessageBox.warning(self, "Chyba", f"Audio nebylo nalezeno.")

    def load_batch_query_from_folder(self):
        word = self.input_batch_query.text().strip()
        if not word: return
        for ext in APP_CFG.supported_extensions:
            p = os.path.join(APP_CFG.offline_db_path, word + ext)
            if os.path.exists(p):
                self.batch_query_path = p
                self.lbl_batch_query.setText(os.path.basename(p)); self.lbl_batch_query.setStyleSheet("color: #10b981; font-weight: bold;")
                self.log_status(f"✅ Vzorek načten z databáze.")
                return
        QMessageBox.warning(self, "Chyba", f"Slovo nebylo nalezeno.")

    def select_batch_query(self):
        path, _ = QFileDialog.getOpenFileName(self, "Vyberte vzorek hledaného slova", "", "Audio (*.mp3 *.wav *.m4a *.flac)")
        if path:
            self.batch_query_path = path
            self.lbl_batch_query.setText(os.path.basename(path)); self.lbl_batch_query.setStyleSheet("color: #10b981; font-weight: bold;")
            self.input_batch_query.setText(os.path.splitext(os.path.basename(path))[0])

    def log_batch_append(self, text):
        self.log_batch.append(text)
        sb = self.log_batch.verticalScrollBar(); sb.setValue(sb.maximum())

    def run_batch_benchmark(self):
        if not getattr(self, 'batch_long_path', None) or not getattr(self, 'batch_query_path', None):
            QMessageBox.warning(self, "Chyba", "Nejprve vyberte prohledávané dlouhé audio a vzorek slova.")
            return
            
        gt_text = self.combo_gt.currentText().strip()
        if os.path.exists(gt_text):
            self.ground_truth_path = gt_text
        else:
            possible_path = os.path.join(os.path.dirname(APP_CFG.offline_db_path), gt_text)
            if os.path.exists(possible_path): self.ground_truth_path = possible_path
            else:
                QMessageBox.warning(self, "Chybí Standard", "Vybraný zlatý standard neexistuje.")
                return
                
        self.txt_batch_detail.clear(); self.log_batch.clear(); self.table_batch.setRowCount(0)
        self.btn_run_batch.setEnabled(False); self.combo_batch_method.setEnabled(False)
        self.btn_batch_pause.setEnabled(True); self.btn_batch_stop.setEnabled(True)
        
        self.worker_batch = BatchEvaluationWorker(self.batch_long_path, self.batch_query_path, self.ground_truth_path, self.combo_batch_method.currentData(), float(self.input_threshold.text()))
        self.worker_batch.progress.connect(lambda v, m: (self.progress_batch.setValue(v), self.lbl_batch_status.setText(m)))
        self.worker_batch.log_msg.connect(self.log_batch_append)
        self.worker_batch.result_row.connect(self.on_batch_row)
        self.worker_batch.finished.connect(self.on_batch_finished) 
        self.worker_batch.start()

    def on_batch_finished(self):
        self.btn_run_batch.setEnabled(True); self.combo_batch_method.setEnabled(True)
        self.btn_batch_pause.setEnabled(False); self.btn_batch_stop.setEnabled(False)
        self.btn_batch_pause.setText("Pauza")

    def on_batch_row(self, data):
        self.batch_detailed_data[data['word']] = data
        row = self.table_batch.rowCount()
        self.table_batch.insertRow(row)
        
        cell_widget = QWidget(); cell_layout = QHBoxLayout(cell_widget); cell_layout.setContentsMargins(5, 2, 5, 2)
        lbl_word = QLabel(data['word']); lbl_word.setStyleSheet("font-weight: bold;")
        btn_play = QPushButton(); btn_play.setFixedSize(30, 24)
        if os.path.exists(APP_CFG.audio_icon_path): btn_play.setIcon(QIcon(APP_CFG.audio_icon_path))
        else: btn_play.setText("🔊")
        btn_play.setStyleSheet("QPushButton { background-color: transparent; border: none; }")
        btn_play.clicked.connect(lambda checked, w=data['word']: self.play_exhaustive_word(w))
        cell_layout.addWidget(lbl_word); cell_layout.addStretch(); cell_layout.addWidget(btn_play)
        self.table_batch.setCellWidget(row, 0, cell_widget)

        tp = data.get('tp', 0); fp = data.get('fp', 0); fn = data.get('gt_count', 0) - tp
        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision/100 * recall/100) / (precision/100 + recall/100) if (precision + recall) > 0 else 0.0
        
        hodnoty = [str(data.get('gt_count', 0)), str(data.get('found_count', 0)), str(tp), str(fp), str(fn), f"{precision:.1f}%", f"{recall:.1f}%", f"{data.get('frr', 0) * 100:.1f}%", f"{data.get('fa_h', 0):.2f}", f"{f1:.3f}"]
        for col_idx, hodnota in enumerate(hodnoty, start=1):
            item = QTableWidgetItem(hodnota); item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table_batch.setItem(row, col_idx, item)

    def on_batch_row_selected(self):
        items = self.table_batch.selectedItems()
        if not items: return
        widget = self.table_batch.cellWidget(items[0].row(), 0)
        if not widget: return
        word = widget.findChild(QLabel).text()
        data = self.batch_detailed_data.get(word)
        if not data: return

        gt_times_str = ", ".join([f"{t}s" for t in data.get('gt_times', [])]) if data.get('gt_times') else "Žádný výskyt"
        app_times_str = ", ".join([f"{t}s" for t in data.get('found_times', [])]) if data.get('found_times') else "Nic nenalezeno"
        self.txt_batch_detail.setHtml(f"""<h3 style="color: #10b981;">Slovo: '{word.upper()}'</h3><p><b>Skutečné časy:</b><br><span style="color: #4b5563;">{gt_times_str}</span></p><p><b>Časy nalezené aplikací:</b><br><span style="color: #4b5563;">{app_times_str}</span></p>""")

    def play_exhaustive_word(self, word):
        data = self.batch_detailed_data.get(word)
        if not data or not data.get("found_times", []): return

        times = data["found_times"]
        timestamp_s = times[0] if len(times) == 1 else None
        if len(times) > 1:
            items = [f"Čas: {t} s" for t in times]
            item, ok = QInputDialog.getItem(self, f"Přehrát '{word}'", f"Vyberte čas:", items, 0, False)
            if ok and item: timestamp_s = times[items.index(item)]

        if timestamp_s is not None:
            try:
                y, sr = librosa.load(self.batch_long_path, sr=16000)
                start_sample = int(timestamp_s * sr); pad = int(0.05 * sr)
                self.play_mono(y[max(0, start_sample-pad) : min(len(y), start_sample + int(1.0 * sr)+pad)], sr, f"Kvantitativní nález: {word} ({timestamp_s}s)", "E")
            except Exception as e: self.log_status(f"❌ Chyba při přehrávání: {e}")

    def select_file(self, type_):
        path, _ = QFileDialog.getOpenFileName(self, "Vyber audio", "", "Audio (*.mp3 *.wav *.m4a *.flac)")
        if not path: return
        name = os.path.basename(path)
        
        if type_ == 'long_single': 
            self.path_long_single = path; self.input_single_long.setText(name)
            self.lbl_single_long.setText(name); self.lbl_single_long.setStyleSheet("color: #10b981; font-weight: bold;")
        elif type_ == 'query_single': 
            self.path_query = path; self.input_query.setText(name)
            self.lbl_single_query.setText(name); self.lbl_single_query.setStyleSheet("color: #10b981; font-weight: bold;")
        elif type_ == 'long_corpus': 
            self.path_long_corpus = path; self.input_corpus_long.setText(name)
            self.lbl_corpus_long.setText(name); self.lbl_corpus_long.setStyleSheet("color: #10b981; font-weight: bold;")
            self.display_ground_truth_text(path)

    def load_single_long_from_folder(self):
        audio_name = self.input_single_long.text().strip()
        if not audio_name: return
        if audio_name.isdigit():
            audio_name = f"sample-{int(audio_name):06d}"
            self.input_single_long.setText(audio_name)
            
        base_dir = os.path.dirname(APP_CFG.offline_db_path)
        for d in [APP_CFG.offline_db_path, base_dir, os.path.join(base_dir, "cv_16k_wav"), os.path.join(base_dir, "Common Voice Spontaneous Speech 3.0 - English")]:
            for ext in APP_CFG.supported_extensions:
                p = os.path.join(d, audio_name + ext)
                if os.path.exists(p):
                    self.path_long_single = p
                    self.lbl_single_long.setText(os.path.basename(p)); self.lbl_single_long.setStyleSheet("color: #10b981; font-weight: bold;")
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
                self.lbl_single_query.setText(os.path.basename(p)); self.lbl_single_query.setStyleSheet("color: #10b981; font-weight: bold;")
                self.log_status(f"✅ Vzorek '{word}' dohledán.")
                return
        QMessageBox.warning(self, "Chyba", f"Slovo '{word}' nebylo nalezeno.")

    def toggle_single_pause(self):
        if hasattr(self, 'worker1') and self.worker1.isRunning():
            self.btn_single_pause.setText("Pokračovat" if self.worker1.toggle_pause() else "Pauza")

    def stop_single_scan(self):
        if hasattr(self, 'worker1') and self.worker1.isRunning():
            self.log_text.append("⚠️ Ukončuji analýzu..."); QApplication.processEvents()
            self.btn_single_pause.setEnabled(False); self.btn_single_stop.setEnabled(False)
            self.worker1.stop()
            if not self.worker1.wait(500):
                self.worker1.terminate(); self.worker1.wait()
                self.on_single_finished(MatchResult(score=0, start_f=0, end_f=0, method="", error="Zrušeno uživatelem."))

    def toggle_batch_pause(self):
        if hasattr(self, 'worker_batch') and self.worker_batch.isRunning():
            self.btn_batch_pause.setText("Pokračovat" if self.worker_batch.toggle_pause() else "Pauza")

    def stop_batch_scan(self):
        if hasattr(self, 'worker_batch') and self.worker_batch.isRunning():
            self.log_batch.append("⚠️ Ukončuji evaluaci..."); QApplication.processEvents()
            self.btn_batch_pause.setEnabled(False); self.btn_batch_stop.setEnabled(False)
            self.worker_batch.stop()
            if not self.worker_batch.wait(500):
                self.worker_batch.terminate(); self.worker_batch.wait()
                self.on_batch_finished()
                
    def toggle_corpus_pause(self):
        if hasattr(self, 'worker2') and self.worker2.isRunning():
            is_paused = self.worker2.toggle_pause()
            self.btn_corpus_pause.setText("POKRAČOVAT" if is_paused else "Pauza")
            self.btn_corpus_pause.setStyleSheet("background-color: #fef3c7; color: #92400e; font-weight: bold;" if is_paused else "")

    def stop_corpus_scan(self):
        if hasattr(self, 'worker2') and self.worker2.isRunning():
            self.log_corpus.append("⚠️ Probíhá okamžité ukončování analýzy..."); QApplication.processEvents() 
            self.btn_corpus_pause.setEnabled(False); self.btn_corpus_stop.setEnabled(False)
            self.worker2.stop()
            if not self.worker2.wait(500):
                self.worker2.terminate(); self.worker2.wait() 
                self.on_corpus_scan_finished({"type": "error", "msg": "Analýza byla přerušena uživatelem."})

    def run_single_analysis(self):
        self.log_text.clear(); self.single_found_ranges = [] 
        self.start_single_worker()

    def run_single_next(self): 
        self.log_status("Pokračuji v hledání dalšího výskytu...")
        self.start_single_worker()

    def start_single_worker(self):
        self.btn_analyze.setEnabled(False); self.combo_method.setEnabled(False)
        self.btn_single_pause.setEnabled(True); self.btn_single_stop.setEnabled(True)
        self.btn_next.setEnabled(False); self.btn_visual.setEnabled(False)
        for btn in [self.btn_listen_L, self.btn_listen_stereo, self.btn_listen_R]: btn.setEnabled(False)
            
        self.progress_detail.setValue(0)
        
        if not getattr(self, 'path_long_single', None) or not getattr(self, 'path_query', None):
            self.log_text.append("❌ Chyba: Chybí cesta k audiu nebo vzorku.")
            self.on_single_finished(MatchResult(score=0, start_f=0, end_f=0, method="", error="Chyba vstupů"))
            return

        method = self.combo_method.currentData()
        self.log_status(f"Startuji proces analýzy (Metoda: {method})...")

        self.worker1 = SingleAnalysisWorker(self.path_long_single, self.path_query, method, self.single_found_ranges)
        self.worker1.progress.connect(self.progress_detail.setValue)
        self.worker1.finished.connect(self.on_single_finished)
        self.worker1.start()

    def on_single_finished(self, res: MatchResult):
        self.btn_analyze.setEnabled(True); self.combo_method.setEnabled(True)
        self.btn_single_pause.setEnabled(False); self.btn_single_stop.setEnabled(False); self.btn_single_pause.setText("Pauza")
        self.progress_detail.setValue(100)
        
        if not res.is_success:
            if "(již) nalezeno" in (res.error or ""):
                self.log_text.append("Hledání dokončeno.")
                self.show_msg("Konec hledání", "(již) nalezeno", QMessageBox.Icon.Information)
            elif "Zrušeno" in (res.error or ""):
                self.log_text.append(f"❌ {res.error}")
            else:
                self.log_text.append(f"\n❌ DETAILNÍ VÝPIS CHYBY:\n{res.error}\n")
                self.show_msg("Kritická chyba", "Aplikace spadla. Podívejte se do konzole dole na celý výpis chyby.", QMessageBox.Icon.Critical)
            return
            
        self.log_text.append(f"✅ Nalezeno! Čas: {res.start_f / (res.sr / AUDIO_CFG.hop_length):.2f} s")
        self.single_current_result = res
        self.single_found_ranges.append((res.start_f, res.end_f))

        self.btn_next.setEnabled(True); self.btn_visual.setEnabled(True)
        for btn in [self.btn_listen_L, self.btn_listen_stereo, self.btn_listen_R]: btn.setEnabled(True)
        
        try: self.draw_results(res, self.canvas_single, self.single_found_ranges)
        except Exception as e: self.log_text.append(f"❌ Chyba při vykreslování: {e}")

    def update_corpus_progress(self, val, msg):
        self.progress_corpus.setValue(val)
        if msg:
            self.log_corpus.append(f"{msg}")
            sb = self.log_corpus.verticalScrollBar(); sb.setValue(sb.maximum())

    def run_corpus_scan(self):
        if not getattr(self, 'path_long_corpus', None):
            QMessageBox.warning(self, "Chyba", "Nejprve vyberte dlouhé audio."); return
            
        self.btn_corpus_scan.setEnabled(False); self.combo_corpus_method.setEnabled(False)
        self.btn_corpus_pause.setEnabled(True); self.btn_corpus_stop.setEnabled(True)
        self.log_corpus.clear(); self.progress_corpus.setValue(0); self.table.setRowCount(0)
        
        self.worker2 = CorpusScannerWorker(self.path_long_corpus, APP_CFG.database_file, self.combo_corpus_method.currentData())
        self.worker2.progress.connect(self.update_corpus_progress)
        self.worker2.finished.connect(self.on_corpus_scan_finished)
        self.worker2.start()

    def on_corpus_scan_finished(self, res):
        self.btn_corpus_scan.setEnabled(True); self.combo_corpus_method.setEnabled(True)
        self.btn_corpus_pause.setEnabled(False); self.btn_corpus_stop.setEnabled(False)
        self.btn_corpus_pause.setText("Pozastavení analýzy"); self.btn_corpus_pause.setStyleSheet("")
        self.progress_corpus.setValue(100)
        
        if res["type"] == "error": 
            self.log_corpus.append(f"❌ Chyba: {res.get('msg', 'Neznámá chyba')}")
            return

        whisper_text = res.get('whisper_text', '')
        if whisper_text:
            self.lbl_whisper_text.setText(f"<b>Surový přepis Whisperu:</b> {whisper_text}")
            self.lbl_whisper_text.show(); self.frame_corpus_gt.show()
            
        self.log_corpus.append("✅ Analýza úspěšně dokončena! Vykresluji výsledky...")
        
        all_results = res['results']
        MAX_SAFE = 1500
        if len(all_results) > MAX_SAFE:
            self.log_corpus.append(f"⚠ Pozor: Nalezeno {len(all_results)} shod! Vykresluji {MAX_SAFE} nejlepších.")
            all_results = sorted(all_results, key=lambda x: x['score'], reverse=(self.combo_corpus_method.currentData() == 'pattern'))[:MAX_SAFE]

        final_results = sorted(all_results, key=lambda x: x['start_f'])
        self.corpus_data = res
        self.corpus_data['results'] = final_results 
        self.table.setRowCount(len(final_results))
        
        for i, row in enumerate(final_results):
            self.corpus_history[row['word']] = [(row['start_f'], row['end_f'])]
            
            cell_widget = QWidget(); cell_layout = QHBoxLayout(cell_widget); cell_layout.setContentsMargins(5, 2, 5, 2)
            lbl_word = QLabel(row['word']); lbl_word.setStyleSheet("font-weight: bold;")
            btn_play = QPushButton(); btn_play.setFixedSize(30, 24); btn_play.setCursor(Qt.CursorShape.PointingHandCursor)
            if os.path.exists(APP_CFG.audio_icon_path): btn_play.setIcon(QIcon(APP_CFG.audio_icon_path))
            else: btn_play.setText("🔊")
            btn_play.setStyleSheet("QPushButton { background-color: transparent; border: none; } QPushButton:hover { background-color: rgba(16, 185, 129, 0.2); border-radius: 4px; }")
            btn_play.clicked.connect(lambda checked, idx=i: self.play_specific_corpus_word(idx))
            
            cell_layout.addWidget(lbl_word); cell_layout.addStretch(); cell_layout.addWidget(btn_play)
            self.table.setCellWidget(i, 0, cell_widget)
            self.table.setItem(i, 1, QTableWidgetItem(f"{row['score']:.4f}"))
            self.table.setItem(i, 2, QTableWidgetItem(f"{row['start_f'] * AUDIO_CFG.hop_length / res['sr']:.1f}"))

    def on_corpus_method_changed(self):
        self.lbl_whisper_text.hide(); self.lbl_whisper_text.setText("")
        if not self.lbl_corpus_gt.text(): self.frame_corpus_gt.hide()

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
            self.log_corpus.append("Načítám audio do paměti pro přehrávání..."); QApplication.processEvents()
            y_audio, sr = librosa.load(self.path_long_corpus, sr=sr)
            self.corpus_data['y_long'] = y_audio 

        self.play_mono(y_audio[start_sample:min(len(y_audio), end_sample + padding_samples)], sr, f"Slovo z korpusu: '{word}'", "C")

    def draw_results(self, res: MatchResult, canvas, found_ranges):
        canvas.figure.clear()
        ax1, ax2 = canvas.figure.subplots(1, 2)
        ax1.set_facecolor('#000000'); ax2.set_facecolor('#000000')
        
        frames_per_sec = res.sr / AUDIO_CFG.hop_length
        target_frames = int(10.0 * frames_per_sec)
        
        def pad_with_min(array, pad_l, pad_r, min_val):
            return np.pad(array, ((0, 0), (max(0, pad_l), max(0, pad_r))), mode='constant', constant_values=min_val) if pad_l > 0 or pad_r > 0 else array

        num_mels = res.vis_spec_query.shape[0]
        y_ticks, x_ticks_pos = np.arange(0, num_mels + 1, 16), np.arange(0, 11, 1)

        query_frames = res.vis_spec_query.shape[1]
        if query_frames < target_frames:
            spec_query_10s = pad_with_min(res.vis_spec_query, (target_frames - query_frames) // 2, (target_frames - query_frames) - ((target_frames - query_frames) // 2), np.min(res.vis_spec_query))
        else:
            spec_query_10s = res.vis_spec_query[:, (query_frames - target_frames) // 2 : ((query_frames - target_frames) // 2) + target_frames]
            
        librosa.display.specshow(spec_query_10s, ax=ax1, x_axis='s', sr=res.sr, hop_length=AUDIO_CFG.hop_length)
        ax1.set_title("Hledaný vzorek"); ax1.set_xlabel("Čas vzorku (s)")
        ax1.set_ylabel("Mel pásmo (Index)"); ax1.set_yticks(y_ticks); ax1.tick_params(axis='y', labelleft=True)
        ax1.set_ylim(0, num_mels); ax1.set_xticks(x_ticks_pos); ax1.set_xlim(0, 10.0); ax1.margins(x=0, y=0)
        
        start_s = res.start_f / frames_per_sec
        page_start_s = float(int(start_s // 10) * 10)
        page_start_frames = int(page_start_s * frames_per_sec)
        page_end_frames = page_start_frames + target_frames
        
        max_frames = res.vis_spec_long.shape[1]
        valid_slice = res.vis_spec_long[:, max(0, page_start_frames):min(max_frames, page_end_frames)]
        
        img2 = librosa.display.specshow(pad_with_min(valid_slice, max(0, -page_start_frames), max(0, page_end_frames - max_frames), np.min(res.vis_spec_long)), ax=ax2, x_axis='s', sr=res.sr, hop_length=AUDIO_CFG.hop_length)
        
        ax2.set_title(f"Detail nálezu | Skutečný čas v nahrávce: {start_s:.2f} s"); ax2.set_xlabel("Absolutní čas nahrávky (s)")
        ax2.set_ylabel(""); ax2.set_yticks(y_ticks); ax2.tick_params(axis='y', labelleft=True)
        ax2.set_ylim(0, num_mels); ax2.margins(x=0, y=0)
        ax2.set_xticks(x_ticks_pos); ax2.set_xticklabels([str(int(page_start_s + i)) for i in x_ticks_pos]); ax2.set_xlim(0, 10.0)
        
        for i, (f_start, f_end) in enumerate(found_ranges):
            s_start, s_end = f_start / frames_per_sec, f_end / frames_per_sec
            if s_end > page_start_s and s_start < page_start_s + 10.0:
                ax2.axvspan(max(0.0, s_start - page_start_s), min(10.0, s_end - page_start_s), color='#10b981' if i == len(found_ranges) - 1 else '#9ca3af', alpha=0.4 if i == len(found_ranges) - 1 else 0.5)

        canvas.figure.colorbar(img2, ax=ax2, format="%+2.0f dB", fraction=0.046, pad=0.04)
        canvas.figure.subplots_adjust(left=0.06, right=0.92, bottom=0.12, top=0.90, wspace=0.10) 
        canvas.draw()

    def create_listen_buttons(self, play_func):
        btn_L = QPushButton("L (Vzorek)"); btn_L.clicked.connect(lambda: play_func('left'))
        btn_S = QPushButton("Přehrát (Stereo)"); btn_S.clicked.connect(lambda: play_func('stereo'))
        btn_R = QPushButton("P (Nález)"); btn_R.clicked.connect(lambda: play_func('right'))
        return btn_L, btn_S, btn_R

    def play_stereo_match(self, y_query: np.ndarray, y_match: np.ndarray, sr: int):
        if y_query is None or y_match is None or len(y_query) == 0 or len(y_match) == 0: return

        target_sr = 48000
        y_q = np.concatenate((np.zeros(int(0.2 * target_sr), dtype=np.float32), librosa.resample(librosa.util.normalize(y_query), orig_sr=sr, target_sr=target_sr)))
        y_m = np.concatenate((np.zeros(int(0.2 * target_sr), dtype=np.float32), librosa.resample(librosa.util.normalize(y_match), orig_sr=sr, target_sr=target_sr)))

        max_len = max(len(y_q), len(y_m))
        self._execute_windows_native_playback(np.vstack([np.pad(y_q, (0, max_len - len(y_q))), np.pad(y_m, (0, max_len - len(y_m)))]).T, target_sr, "stereo")

    def play_mono(self, y_audio: np.ndarray, sr: int, log_msg: str, side: str):
        if y_audio is None or len(y_audio) == 0: return
        target_sr = 48000
        y_final = np.concatenate((np.zeros(int(0.2 * target_sr), dtype=np.float32), librosa.resample(librosa.util.normalize(y_audio), orig_sr=sr, target_sr=target_sr)))
        if hasattr(self, 'log_text') and self.log_text.isVisible(): self.log_text.append(f"▶ Přehrávám: {log_msg}")
        self._execute_windows_native_playback(np.vstack((y_final, y_final)).T, target_sr, f"mono_{side}")

    def _execute_windows_native_playback(self, signal: np.ndarray, sr: int, label: str):
        try:
            tmp_file = os.path.join(tempfile.gettempdir(), f"shazam_native_{label}.wav")
            sf.write(tmp_file, signal, sr, subtype='PCM_16')
            winsound.PlaySound(None, winsound.SND_PURGE)
            winsound.PlaySound(tmp_file, winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception as e:
            if hasattr(self, 'log_text') and self.log_text.isVisible(): self.log_text.append(f"❌ Chyba nativního přehrávání: {e}")
                
    def play_single(self, mode: str):
        if not getattr(self, 'single_current_result', None): return
        res = self.single_current_result
        padding_samples = int(APP_CFG.playback_padding_sec * res.sr)
        y_match = res.y_long[max(0, int(res.start_f * AUDIO_CFG.hop_length) - padding_samples):min(len(res.y_long), int(res.end_f * AUDIO_CFG.hop_length) + padding_samples)]

        if mode == 'stereo': self.play_stereo_match(res.y_query, y_match, res.sr)
        elif mode == 'left': self.play_mono(res.y_query, res.sr, "Hledaný vzorek", "L")
        elif mode == 'right': self.play_mono(y_match, res.sr, "Detail nálezu", "P")
    
    def open_settings(self): SettingsDialog(self).exec()
        
    def open_visual_comparison(self, res):
        if res: VisualCompareDialog(res, self).exec()
        
    def apply_native_titlebar_color(self, target_widget=None, hex_bg="#10b981", hex_text="#ffffff"):
        if sys.platform != "win32": return
        try:
            hwnd = HWND(int((target_widget if target_widget else self).winId()))
            bg, txt = hex_bg.lstrip('#'), hex_text.lstrip('#')
            ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, DWORD(35), ctypes.byref(DWORD((int(bg[4:6], 16) << 16) | (int(bg[2:4], 16) << 8) | int(bg[0:2], 16))), ctypes.sizeof(DWORD))
            ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, DWORD(36), ctypes.byref(DWORD((int(txt[4:6], 16) << 16) | (int(txt[2:4], 16) << 8) | int(txt[0:2], 16))), ctypes.sizeof(DWORD))
        except Exception: pass

    def show_msg(self, title, text, icon=QMessageBox.Icon.Information):
        msg = QMessageBox(self); msg.setWindowTitle(title); msg.setText(text); msg.setIcon(icon)
        self.apply_native_titlebar_color(target_widget=msg)
        msg.exec()