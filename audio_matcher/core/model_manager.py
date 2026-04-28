# ==============================================================================
# audio_matcher/core/model_manager.py
# ==============================================================================
import threading
import torch
import warnings
from config import MODEL_CFG

class ModelManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.w2v_processor = None
                    cls._instance.w2v_model = None
                    cls._instance.whisper_model = None
                    cls._instance.silero_model = None
                    cls._instance.silero_utils = None
        return cls._instance

    def get_device(self):
        return torch.device(MODEL_CFG.device if torch.cuda.is_available() and MODEL_CFG.device == "cuda" else "cpu")

    def get_whisper(self):
        if self.whisper_model is None:
            with self._lock:
                if self.whisper_model is None:
                    warnings.filterwarnings("ignore", message="Failed to launch Triton kernels")
                    try:
                        import whisper
                        device_str = "cuda" if torch.cuda.is_available() and MODEL_CFG.device == "cuda" else "cpu"
                        self.whisper_model = whisper.load_model("small", device=device_str)
                        if device_str == "cuda":
                            self.whisper_model = self.whisper_model.to("cuda")
                    except ImportError:
                        return None
        return self.whisper_model

    def get_wav2vec(self):
        if self.w2v_processor is None:
            with self._lock:
                if self.w2v_processor is None:
                    warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
                    from transformers import Wav2Vec2Processor, Wav2Vec2Model
                    self.w2v_processor = Wav2Vec2Processor.from_pretrained(MODEL_CFG.wav2vec_local_path)
                    self.w2v_processor.tokenizer_class = "Wav2Vec2CTCTokenizer"
                    self.w2v_model = Wav2Vec2Model.from_pretrained(MODEL_CFG.wav2vec_local_path).to(self.get_device())
        return self.w2v_processor, self.w2v_model

    def get_silero_vad(self):
        if self.silero_model is None:
            with self._lock:
                if self.silero_model is None:
                    from core.utils_vad import get_speech_timestamps
                    self.silero_model = torch.jit.load(MODEL_CFG.silero_vad_local_path).to(self.get_device())
                    self.silero_model.eval()
                    self.silero_utils = [get_speech_timestamps]
        return self.silero_model, self.silero_utils