import numpy as np
from scipy import signal

def apply_filter(audio_data, sample_rate, filter_type, cutoff_freq=None, 
                 low_cut=None, high_cut=None, order=4):
    """
    Aplica filtro IIR a los datos del audio usando Butterworth!
    """
    nyquist = 0.5 * sample_rate
    
    if filter_type == 'lowpass':
        normal_cutoff = cutoff_freq / nyquist
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        filtered = signal.lfilter(b, a, audio_data)
    
    elif filter_type == 'highpass':
        normal_cutoff = cutoff_freq / nyquist
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        filtered = signal.lfilter(b, a, audio_data)
    
    elif filter_type == 'bandpass':
        low = low_cut / nyquist
        high = high_cut / nyquist
        b, a = signal.butter(order, [low, high], btype='band')
        filtered = signal.lfilter(b, a, audio_data)
    
    elif filter_type == 'bandstop':
        low = low_cut / nyquist
        high = high_cut / nyquist
        b, a = signal.butter(order, [low, high], btype='bandstop')
        filtered = signal.lfilter(b, a, audio_data)
    
    else:
        raise ValueError("Tipo de filtro falopa")
    
    return filtered