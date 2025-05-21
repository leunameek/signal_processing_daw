import numpy as np
from scipy import signal
import soundfile as sf

# decidimos generar el impulso con mates para q suene mejor, el globo sonaba muy feo jeje

def generate_impulse_response(sample_rate, decay_time=1.0, decay_factor=0.5):
    """Genera el impulso más pulido"""
    length = int(sample_rate * decay_time)
    t = np.linspace(0, decay_time, length)
    impulse = np.random.randn(length) * np.exp(-t / decay_factor)
    return impulse / np.max(np.abs(impulse))

def apply_reverb(audio_data, sample_rate, impulse_response=None, mix=0.3, decay_time=1.0):
    """
    Apply reverb using convolution
    mix: 0 (dry) to 1 (wet)
    """
    if impulse_response is None:
        impulse_response = generate_impulse_response(sample_rate, decay_time)
    
    audio_data = np.squeeze(audio_data)
    impulse_response = np.squeeze(impulse_response)
    
    # nuestra querida fft la usamos en la convolucion
    wet_signal = signal.fftconvolve(audio_data, impulse_response, mode='same')
    
    # normaliza y  mezcla con dry la señal
    wet_signal = wet_signal / np.max(np.abs(wet_signal)) * np.max(np.abs(audio_data))
    output = (1 - mix) * audio_data + mix * wet_signal
    
    return output