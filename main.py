import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QPushButton, QSlider, QLabel, QFileDialog,
                             QMessageBox, QGroupBox)
from PyQt5.QtCore import Qt
import soundfile as sf
import sounddevice as sd
from scipy.signal import spectrogram
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from audio_processor import AudioProcessor
from effects.filters import apply_filter
from effects.reverb import apply_reverb

class SimpleDAW(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DAWsito - Procesamiento de Señales")
        self.setGeometry(100, 100, 1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        main_layout.addWidget(control_widget)

        self.plot_canvas = FigureCanvas(Figure(figsize=(12, 6)))
        fig = self.plot_canvas.figure
        self.ax_waveform_original = fig.add_subplot(211)
        self.ax_waveform_processed = fig.add_subplot(212)


        plot_container = QVBoxLayout()
        plot_container.addWidget(QLabel("Visualización De La Señal"))
        plot_container.addWidget(self.plot_canvas)

        plot_widget = QWidget()
        plot_widget.setLayout(plot_container)
        main_layout.addWidget(plot_widget)

        main_layout.setStretch(0, 2)
        main_layout.setStretch(1, 3)

        self.file_info_label = QLabel("No hay audio!")
        self.file_info_label.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(self.file_info_label)

        btn_layout = QHBoxLayout()
        self.load_button = QPushButton("Cargar Audio")
        self.save_button = QPushButton("Guardar Audio Procesao")
        self.reset_button = QPushButton("Deshacer Procesamiento")
        btn_layout.addWidget(self.load_button)
        btn_layout.addWidget(self.save_button)
        btn_layout.addWidget(self.reset_button)
        control_layout.addLayout(btn_layout)

        playback_group = QGroupBox("Controles De Reproducción")
        playback_layout = QHBoxLayout()
        self.play_original_btn = QPushButton("Reproducir Original")
        self.play_processed_btn = QPushButton("Reproducir Procesado")
        self.stop_playback_btn = QPushButton("Parar")
        playback_layout.addWidget(self.play_original_btn)
        playback_layout.addWidget(self.play_processed_btn)
        playback_layout.addWidget(self.stop_playback_btn)
        playback_group.setLayout(playback_layout)
        control_layout.addWidget(playback_group)

        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(80)
        self.volume_label = QLabel("Volumen: 80%")
        control_layout.addWidget(QLabel("Volumen de Reproducción:"))
        control_layout.addWidget(self.volume_slider)
        control_layout.addWidget(self.volume_label)

        # Filter Controls
        filters_group = QGroupBox("Filtros del Audio")
        filters_layout = QVBoxLayout()

        self.lowpass_slider = QSlider(Qt.Horizontal)
        self.lowpass_slider.setRange(20, 20000)
        self.lowpass_slider.setValue(20000)
        self.lowpass_label = QLabel("Corte De Pasa Bajas: 20000 Hz")
        self.apply_lowpass = QPushButton("Aplicar Pasa Bajas")
        filters_layout.addWidget(QLabel("Filtro Pasa Bajas:"))
        filters_layout.addWidget(self.lowpass_slider)
        filters_layout.addWidget(self.lowpass_label)
        filters_layout.addWidget(self.apply_lowpass)

        self.highpass_slider = QSlider(Qt.Horizontal)
        self.highpass_slider.setRange(20, 20000)
        self.highpass_slider.setValue(20)
        self.highpass_label = QLabel("Corte de Pasa Altas: 20 Hz")
        self.apply_highpass = QPushButton("Aplicar Pasa Altas")
        filters_layout.addWidget(QLabel("Filtro Pasa Altas:"))
        filters_layout.addWidget(self.highpass_slider)
        filters_layout.addWidget(self.highpass_label)
        filters_layout.addWidget(self.apply_highpass)

        self.bandpass_low_slider = QSlider(Qt.Horizontal)
        self.bandpass_low_slider.setRange(20, 20000)
        self.bandpass_low_slider.setValue(300)
        self.bandpass_high_slider = QSlider(Qt.Horizontal)
        self.bandpass_high_slider.setRange(20, 20000)
        self.bandpass_high_slider.setValue(3000)
        self.bandpass_label = QLabel("Rango De Pasa Bandas: 300-3000 Hz")
        self.apply_bandpass = QPushButton("Aplicar Pasa Bandas")
        filters_layout.addWidget(QLabel("Filtro Pasa Bandas:"))
        filters_layout.addWidget(QLabel("Corte De Bajas:"))
        filters_layout.addWidget(self.bandpass_low_slider)
        filters_layout.addWidget(QLabel("Corte de Altas:"))
        filters_layout.addWidget(self.bandpass_high_slider)
        filters_layout.addWidget(self.bandpass_label)
        filters_layout.addWidget(self.apply_bandpass)

        self.bandstop_low_slider = QSlider(Qt.Horizontal)
        self.bandstop_low_slider.setRange(20, 20000)
        self.bandstop_low_slider.setValue(300)
        self.bandstop_high_slider = QSlider(Qt.Horizontal)
        self.bandstop_high_slider.setRange(20, 20000)
        self.bandstop_high_slider.setValue(3000)
        self.bandstop_label = QLabel("Suprime Bandas: 300-3000 Hz")
        self.apply_bandstop = QPushButton("Aplicar Suprime Bandas")
        filters_layout.addWidget(QLabel("Filtro Suprime Bandas:"))
        filters_layout.addWidget(QLabel("Corte De Bajas:"))
        filters_layout.addWidget(self.bandstop_low_slider)
        filters_layout.addWidget(QLabel("Corte De Altas:"))
        filters_layout.addWidget(self.bandstop_high_slider)
        filters_layout.addWidget(self.bandstop_label)
        filters_layout.addWidget(self.apply_bandstop)

        filters_group.setLayout(filters_layout)
        control_layout.addWidget(filters_group)

        reverb_group = QGroupBox("Efecto Reverb")
        reverb_layout = QVBoxLayout()
        self.reverb_decay_slider = QSlider(Qt.Horizontal)
        self.reverb_decay_slider.setRange(1, 50)
        self.reverb_decay_slider.setValue(10)
        self.reverb_decay_label = QLabel("Decay: 1.0s")
        self.reverb_mix_slider = QSlider(Qt.Horizontal)
        self.reverb_mix_slider.setRange(0, 100)
        self.reverb_mix_slider.setValue(30)
        self.reverb_mix_label = QLabel("Mix: 30%")
        self.apply_reverb = QPushButton("Aplicar Reverb")
        reverb_layout.addWidget(QLabel("Decay (0.1-5.0s):"))
        reverb_layout.addWidget(self.reverb_decay_slider)
        reverb_layout.addWidget(self.reverb_decay_label)
        reverb_layout.addWidget(QLabel("Dry/Wet Mix (0-100%):"))
        reverb_layout.addWidget(self.reverb_mix_slider)
        reverb_layout.addWidget(self.reverb_mix_label)
        reverb_layout.addWidget(self.apply_reverb)
        reverb_group.setLayout(reverb_layout)
        control_layout.addWidget(reverb_group)

        self.audio_processor = AudioProcessor()
        self.current_volume = 0.8

        self.load_button.clicked.connect(self.load_audio)
        self.save_button.clicked.connect(self.save_audio)
        self.reset_button.clicked.connect(self.reset_audio)
        self.play_original_btn.clicked.connect(self.play_original_audio)
        self.play_processed_btn.clicked.connect(self.play_processed_audio)
        self.stop_playback_btn.clicked.connect(self.stop_audio_playback)
        self.volume_slider.valueChanged.connect(self.update_volume)

        self.lowpass_slider.valueChanged.connect(self.update_lowpass_label)
        self.apply_lowpass.clicked.connect(self.on_apply_lowpass)
        self.highpass_slider.valueChanged.connect(self.update_highpass_label)
        self.apply_highpass.clicked.connect(self.on_apply_highpass)
        self.bandpass_low_slider.valueChanged.connect(self.update_bandpass_label)
        self.bandpass_high_slider.valueChanged.connect(self.update_bandpass_label)
        self.apply_bandpass.clicked.connect(self.on_apply_bandpass)
        self.bandstop_low_slider.valueChanged.connect(self.update_bandstop_label)
        self.bandstop_high_slider.valueChanged.connect(self.update_bandstop_label)
        self.apply_bandstop.clicked.connect(self.on_apply_bandstop)

        self.reverb_decay_slider.valueChanged.connect(self.update_reverb_decay_label)
        self.reverb_mix_slider.valueChanged.connect(self.update_reverb_mix_label)
        self.apply_reverb.clicked.connect(self.on_apply_reverb)

    def update_waveforms(self):
        self.ax_waveform_original.clear()
        self.ax_waveform_processed.clear()

        if self.audio_processor.original_audio is not None:
            audio = self.audio_processor.original_audio
            self.ax_waveform_original.plot(audio, color='blue', linewidth=0.5)
            self.ax_waveform_original.set_title("Original")

        if self.audio_processor.audio_data is not None:
            audio = self.audio_processor.audio_data
            self.ax_waveform_processed.plot(audio, color='green', linewidth=0.5)
            self.ax_waveform_processed.set_title("Procesada")

        self.plot_canvas.draw()

    def update_volume(self, value):
        self.current_volume = value / 100
        self.volume_label.setText(f"Volumen: {value}%")

    def play_audio_with_volume(self, audio_data):
        if audio_data is None:
            raise ValueError("No hay nada para reproducir!")
        audio_with_volume = audio_data * self.current_volume
        sd.stop()
        sd.play(audio_with_volume, self.audio_processor.sample_rate)

    def load_audio(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Abrir archivo de audio", "", "Audio Files (*.wav *.flac *.ogg *.aiff);;All Files (*)")
        if not filepath:
            return
        try:
            self.audio_processor.load_audio(filepath)
            duration = len(self.audio_processor.audio_data) / self.audio_processor.sample_rate
            self.file_info_label.setText(f"Cargado!: {filepath.split('/')[-1]}\nDuración: {duration:.2f}s\nFrecuencia de muestreo: {self.audio_processor.sample_rate}Hz")
            QMessageBox.information(self, "Excelenteeee", "Audio cargado correctamente!")
            self.update_waveforms()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Esto no existe omeeee:\n{str(e)}")

    def save_audio(self):
        if self.audio_processor.audio_data is None:
            QMessageBox.warning(self, "Pilas!", "No hay nada para guardar, sube un audio primero.")
            return
        filepath, _ = QFileDialog.getSaveFileName(self, "Guardar Archivo De Audio", "", "WAV Files (*.wav);;FLAC Files (*.flac);;All Files (*)")
        if not filepath:
            return
        try:
            self.audio_processor.save_audio(filepath)
            QMessageBox.information(self, "Bien!", "El Audio se guardó correctamente")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Esto no se puede guardar:\n{str(e)}")

    def reset_audio(self):
        try:
            if self.audio_processor.reset_audio():
                QMessageBox.information(self, "Success", "Audio reset to original!")
                self.update_waveforms()
            else:
                QMessageBox.warning(self, "Pero que pasa?", "Si no has subido nada, que vas a deshacer?")
        except Exception as e:
            QMessageBox.critical(self, "Paila!", f"Fallamos, no pude:\n{str(e)}")

    def play_original_audio(self):
        try:
            if self.audio_processor.original_audio is None:
                QMessageBox.warning(self, "Pilaaas", "No has cargado nada")
                return
            self.play_audio_with_volume(self.audio_processor.original_audio)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Reproducción fallida:\n{str(e)}")

    def play_processed_audio(self):
        try:
            if self.audio_processor.audio_data is None:
                QMessageBox.warning(self, "Pilas", "No has procesado nada aún o-o")
                return
            self.play_audio_with_volume(self.audio_processor.audio_data)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Reproducción FALLIDA:\n{str(e)}")

    def stop_audio_playback(self):
        try:
            sd.stop()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No pude parar:\n{str(e)}")

    def update_lowpass_label(self, value):
        self.lowpass_label.setText(f"Corte Pasa Bajas: {value} Hz")

    def on_apply_lowpass(self):
        try:
            cutoff = self.lowpass_slider.value()
            self.audio_processor.apply_effect('lowpass', cutoff_freq=cutoff)
            QMessageBox.information(self, "Excelente", "El filtro fue aplicado!")
            self.update_waveforms()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Fallamos aplicando el filtro:\n{str(e)}")

    def update_highpass_label(self, value):
        self.highpass_label.setText(f"Corte Pasa Altas: {value} Hz")

    def on_apply_highpass(self):
        try:
            cutoff = self.highpass_slider.value()
            self.audio_processor.apply_effect('highpass', cutoff_freq=cutoff)
            QMessageBox.information(self, "Excelente", "El filtro fue aplicado!")
            self.update_waveforms()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Fallamos aplicando el filtro:\n{str(e)}")

    def update_bandpass_label(self, value=None):
        low = self.bandpass_low_slider.value()
        high = self.bandpass_high_slider.value()
        self.bandpass_label.setText(f"Rango De Pasa Bandas: {low}-{high} Hz")

    def on_apply_bandpass(self):
        try:
            low_cut = self.bandpass_low_slider.value()
            high_cut = self.bandpass_high_slider.value()
            self.audio_processor.apply_effect('bandpass', low_cut=low_cut, high_cut=high_cut)
            QMessageBox.information(self, "Excelente", "El filtro fue aplicado!")
            self.update_waveforms()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Fallamos aplicando el filtro:\n{str(e)}")

    def update_bandstop_label(self, value=None):
        low = self.bandstop_low_slider.value()
        high = self.bandstop_high_slider.value()
        self.bandstop_label.setText(f"Rango De Suprime Bandas: {low}-{high} Hz")

    def on_apply_bandstop(self):
        try:
            low_cut = self.bandstop_low_slider.value()
            high_cut = self.bandstop_high_slider.value()
            self.audio_processor.apply_effect('bandstop', low_cut=low_cut, high_cut=high_cut)
            QMessageBox.information(self, "Excelente", "El filtro fue aplicado!")
            self.update_waveforms()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Fallamos aplicando el filtro:\n{str(e)}")

    def update_reverb_decay_label(self, value):
        seconds = value / 10
        self.reverb_decay_label.setText(f"Decay: {seconds:.1f}s")

    def update_reverb_mix_label(self, value):
        self.reverb_mix_label.setText(f"Mix: {value}%")

    def on_apply_reverb(self):
        try:
            decay_time = self.reverb_decay_slider.value() / 10
            mix = self.reverb_mix_slider.value() / 100
            self.audio_processor.apply_effect('reverb', decay_time=decay_time, mix=mix)
            QMessageBox.information(self, "Bieeen", "Reverb aplicaaao!")
            self.update_waveforms()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Fallamos Aplicando el reverb:\n{str(e)}")

    def closeEvent(self, event):
        self.stop_audio_playback()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    daw = SimpleDAW()
    daw.show()
    sys.exit(app.exec_())