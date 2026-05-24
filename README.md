
# <img src="assets/ikona.png" alt="Icon" width="45" align="absmiddle"> Audio Analysis Tool

AudioAnalysisTool is a powerful desktop application designed for advanced Keyword Spotting (KWS) in large, unannotated audio corpora. It bridges the gap between raw audio data and efficient semantic search by combining traditional acoustic algorithms (MFCC, DTW) with modern deep learning models (Wav2Vec 2.0, OpenAI Whisper).


## Tech Stack

**Language:** Python 3.x  
**GUI Framework:** PyQt6  
**Audio Processing:** Librosa, Soundfile  
**Machine Learning & AI:** PyTorch, OpenAI Whisper, Wav2Vec 2.0, Silero VAD  
**Data & Math:** NumPy, SciPy, HDF5


## Features

* **Multiple Detection Architectures:** Compare and evaluate approaches like Pattern Matching, MFCC + DTW, Wav2Vec 2.0 + DTW, and Whisper.
* **Smart Filtering:** Integrated Silero VAD to automatically filter out silence and stationary noise, radically reducing computational load.
* **Interactive Visualizations:** Side-by-side Mel-spectrogram comparisons for deep morphological and acoustic inspection.
* **Advanced Playback:** Stereo playback mode (target sample in one ear, detected corpus segment in the other) for instant auditory verification.
* **Quantitative Evaluation Module:** Automated benchmarking using industry-standard KWS metrics (Precision, Recall, F1-score, FRR, FA/h).
* **Massive Data Handling:** Uses Lazy Loading and HDF5 caching to process tens of gigabytes of audio (e.g., Common Voice dataset) without exhausting RAM.


## Installation

Due to the total size of the application and required datasets (approx. 30 GB) and GitHub's file size limits, the main archive has been split into several smaller parts in the Releases section.

To use the tool, you must reassemble these parts:

For Windows users:

    Download all parts (release_part_*) into a single folder.

    Open Command Prompt in that folder.

    Run the following command: copy /b release_part_* AudioAnalysisTool.zip

    Extract the resulting .zip archive into the project directory.

## Usage

The workflow is divided into three primary operational modes, accessible via the main window tabs.

### Quick Start: File Input Methods
Audio files can be loaded using two methods:
1. **Local File Selection:** Click the `Procházet vlastní...` button to select any supported audio file (`.wav`, `.mp3`, `.flac`, etc.) from your local drive.
2. **Database Quick ID:** Enter a sample ID (e.g., `105`) or an english keyword (e.g., `hello`) directly into the text field and press **Enter**. If the entry exists within the database, the application will automatically retrieve and load the corresponding file.

---

## 1. Comparative Analysis (Srovnávací analýza)
This mode enables detailed comparative analysis between a **single query sample (keyword)** and a **continuous audio track**.

### Execution Workflow:
1. Specify the target audio track in the **Prohledávané Audio** field.
2. Specify the target keyword in the **Hledaný Vzorek** field.
3. Select the **Analytický model**:
   * `OpenAI Whisper (ASR)` – Transcribes audio to text and performs exact string matching.
   * `Wav2Vec 2.0 + DTW` – Utilizes a neural network for feature extraction paired with Dynamic Time Warping.
   * `MFCC + DTW` – Traditional acoustic feature matching using Mel-Frequency Cepstral Coefficients and DTW.
   * `Pattern Matching` – Performs 2D template matching on spectrogram visual representations.
4. Click **SPUSTIT ANALÝZU**.

### Post-Analysis Actions:
* **Stereo Playback Verification:** Clicking `Přehrát (Stereo)` routes the query sample to the left audio channel and the detected match to the right channel, allowing for immediate auditory validation.
* **Iterative Search:** If multiple occurrences exist in the recording, clicking `Zobrazit další nález` will bypass the current detection and locate the subsequent instance.
* **Visual Comparison:** Opens a modal window displaying a side-by-side comparative visualization of the query and match spectrograms.

---

## 2. Corpus Analysis (Analýza korpusu)
This mode is designed to scan a target audio track for **all known entries** present in the reference database.

### Execution Workflow:
1. Select the target audio track.
2. Choose the analytical method (e.g., `Whisper + DTW Hybrid` for a combination of ASR transcription and subsequent acoustic verification).
3. Click **ANALYZOVAT KORPUS**.
4. The application will process the entire database vocabulary against the audio and populate the results table.

### Interpreting Results:
* The upper green panel displays the **Reference Transcription (Zlatý standard)** alongside the **Raw Whisper Transcription**, allowing you to evaluate the base model's accuracy.
* The data table lists the detected words, their **Distance Score** (lower values indicate higher similarity for DTW), and the **Timestamp (s)** of the occurrence.
* Click the **Speaker Icon (🔊)** adjacent to any entry to initiate instant playback of the detected segment.

---

## 3. Quantitative Evaluation (Kvantitativní evaluace)
A comprehensive toolset for researchers and analysts to measure algorithmic efficacy against annotated ground truth data (JSON format).

### Execution Workflow:
1. Select the **Zlatý standard** reference file (the application defaults to `virtual_stream_ground_truth_complete_time.json`).
2. Specify the target audio track and query sample.
3. Define the **Maximální práh** (the sensitivity limit for match validation).
4. Click **SPUSTIT KVANTITATIVNÍ EVALUACI**.

### Evaluated Metrics:
The application cross-references algorithmic detections against the ground truth annotations to compute the following statistical indicators:

| Metric | Definition |
| :--- | :--- |
| **GT** | The actual number of target occurrences present in the audio according to human annotation. |
| **Nález** | The total number of detections reported by the algorithm. |
| **TP** (True Positives) | Correctly identified occurrences. |
| **FP** (False Positives) | False alarms (the algorithm reported a match where none exists). |
| **FN** (False Negatives) | Missed detections (the algorithm failed to identify an existing occurrence). |
| **Prec (%)** | Precision; the percentage of correct detections out of all reported detections. |
| **Rec (%)** | Recall / Sensitivity; the percentage of actual occurrences successfully detected. |
| **F1** | The harmonic mean of Precision and Recall, representing overall model accuracy (1.000 indicates perfect precision and recall). |


## Documentation

[Documentation](https://is.muni.cz/auth/th/vdlvx/?fakulta=1421;obdobi=9123;sorter=vedouci;balik=214407)


## Authors

[Vissse](https://github.com/Vissse)


## License

[MIT](https://choosealicense.com/licenses/mit/)

