import soundfile as sf
import numpy as np
from scipy import signal
from effects.filters import apply_filter
from effects.reverb import apply_reverb
import sounddevice as sd

class AudioProcessor:
    def __init__(self):
        self.sample_rate = None
        self.audio_data = None
        self.original_audio = None
        self.is_playing = False
    
    def load_audio(self, filepath):
        """Carga el audio (soporta más formatos)"""
        self.audio_data, self.sample_rate = sf.read(filepath, always_2d=False)
        self.original_audio = self.audio_data.copy()
        
        # convierte a mono para más placer
        if len(self.audio_data.shape) > 1:
            self.audio_data = np.mean(self.audio_data, axis=1)
            self.original_audio = self.audio_data.copy()
        
        # normaliza a [-1, 1] de ser necesario
        if np.issubdtype(self.audio_data.dtype, np.integer):
            self.audio_data = self.audio_data.astype(np.float32) / np.iinfo(self.audio_data.dtype).max
            self.original_audio = self.audio_data.copy()
        
        return self.sample_rate, self.audio_data
    
    def save_audio(self, filepath, data=None):
        """Guarda el audio usando soundfile"""
        if data is None:
            data = self.audio_data
        
        # se asegura de que esté dentro de [-1, 1]
        data = np.clip(data, -1.0, 1.0)
        
        sf.write(filepath, data, self.sample_rate, subtype='PCM_16')
    
    def apply_effect(self, effect_type, **params):
        """Aplica el audio especificado"""
        if self.audio_data is None:
            raise ValueError("No hay audio!")
        
        if effect_type == 'lowpass':
            self.audio_data = apply_filter(self.audio_data, self.sample_rate, 'lowpass', 
                                         cutoff_freq=params.get('cutoff_freq'))
        elif effect_type == 'highpass':
            self.audio_data = apply_filter(self.audio_data, self.sample_rate, 'highpass', 
                                         cutoff_freq=params.get('cutoff_freq'))
        elif effect_type == 'bandpass':
            self.audio_data = apply_filter(self.audio_data, self.sample_rate, 'bandpass', 
                                         low_cut=params.get('low_cut'), 
                                         high_cut=params.get('high_cut'))
        elif effect_type == 'bandstop':
            self.audio_data = apply_filter(self.audio_data, self.sample_rate, 'bandstop', 
                                         low_cut=params.get('low_cut'), 
                                         high_cut=params.get('high_cut'))
        elif effect_type == 'reverb':
            self.audio_data = apply_reverb(self.audio_data, self.sample_rate,
                                         decay_time=params.get('decay_time', 1.0),
                                         mix=params.get('mix', 0.3))
        else:
            raise ValueError(f"Unknown effect type: {effect_type}")
        
        return self.audio_data
    
    def reset_audio(self):
        """Deshace los cambios y deja el original"""
        if self.original_audio is not None:
            self.audio_data = self.original_audio.copy()
            return True
        return False