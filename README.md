
# <img src="assets/ikona.png" alt="Icon" width="45" align="absmiddle"> Audio Analysis Tool

# Audio Analysis Tool

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
    
## Documentation

[Documentation](https://is.muni.cz/auth/th/vdlvx/?fakulta=1421;obdobi=9123;sorter=vedouci;balik=214407)


## Authors

[Vissse](https://github.com/Vissse)


## License

[MIT](https://choosealicense.com/licenses/mit/)

