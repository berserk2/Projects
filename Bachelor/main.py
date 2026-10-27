import sys
import os
import time
import json
import socket
import struct
import re
import math
import subprocess
import atexit
import traceback
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import pyvisa

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QLineEdit, QPushButton, QComboBox,
    QCheckBox, QGroupBox, QTabWidget, QListWidget, QListWidgetItem,
    QTextEdit, QProgressBar, QFileDialog, QMessageBox, QTableWidget,
    QTableWidgetItem
)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QColor


# ==============================================================================
# МОДЕЛИ СОСТОЯНИЯ
# ==============================================================================
@dataclass
class ChannelState:
    enabled: bool = False
    waveform: str = "SIN"
    freq: float = 1e6
    high: float = 1.0
    low: float = -1.0
    width: float = 1e-6
    offset: float = 0.0


@dataclass
class OscilloscopeState:
    ch1_enabled: bool = True
    ch2_enabled: bool = True
    ch1_offset: float = 0.0
    ch2_offset: float = 0.0
    vscale: float = 1.0
    timebase: float = 1e-7
    trigger_level: float = 0.5


@dataclass
class SpectrumAnalyzerState:
    center_freq: float = 1e6
    span: float = 200000.0
    rbw: float = 10000.0
    vbw: float = 10000.0


# ==============================================================================
# АППАРАТНЫЙ ДРАЙВЕР ГЕНЕРАТОРА
# ==============================================================================
class RigolGeneratorDriver:
    def __init__(self, resource):
        self.dev = resource
        self.dev.timeout = 5000

    def safe_write(self, cmd: str):
        try:
            self.dev.write(cmd)
            time.sleep(0.05)
        except Exception as e:
            raise RuntimeError(f"Сбой записи в генератор [{cmd}]: {str(e)}")

    def safe_query(self, cmd: str) -> str:
        try:
            res = self.dev.query(cmd)
            time.sleep(0.05)
            return res.strip()
        except Exception as e:
            raise RuntimeError(f"Сбой запроса к генератору [{cmd}]: {str(e)}")

    def set_output(self, ch: int, state: bool):
        state_str = "ON" if state else "OFF"
        self.safe_write(f':OUTP{ch} {state_str}')

    def apply_channel_state(self, ch: int, state: ChannelState):
        self.safe_write(f':SOUR{ch}:FUNC {state.waveform}')
        if state.waveform == "DC":
            self.safe_write(f':SOUR{ch}:VOLT:OFFS {state.offset}')
            self.safe_write(f':SOUR{ch}:VOLT 0')
        elif state.waveform == "PULS":
            self.safe_write(f':SOUR{ch}:FREQ {state.freq}')
            self.safe_write(f':SOUR{ch}:VOLT:HIGH {state.high}')
            self.safe_write(f':SOUR{ch}:VOLT:LOW {state.low}')
            self.safe_write(f':SOUR{ch}:PULS:WIDT {state.width}')
        else:
            self.safe_write(f':SOUR{ch}:FREQ {state.freq}')
            self.safe_write(f':SOUR{ch}:VOLT:HIGH {state.high}')
            self.safe_write(f':SOUR{ch}:VOLT:LOW {state.low}')

        self.set_output(ch, state.enabled)

    def load_state_from_json(self, json_file: str):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for ch_key, ch_num in [("channel1", 1), ("channel2", 2)]:
            ch = data.get(ch_key, {})
            wf = ch.get("waveform", "SIN")
            self.safe_write(f':SOUR{ch_num}:FUNC {wf}')
            if wf == "DC":
                if "offset_v" in ch:
                    self.safe_write(f':SOUR{ch_num}:VOLT:OFFS {ch["offset_v"]}')
                    self.safe_write(f':SOUR{ch_num}:VOLT 0')
            else:
                if "frequency_hz" in ch:
                    self.safe_write(f':SOUR{ch_num}:FREQ {ch["frequency_hz"]}')
                if "high_v" in ch:
                    self.safe_write(f':SOUR{ch_num}:VOLT:HIGH {ch["high_v"]}')
                if "low_v" in ch:
                    self.safe_write(f':SOUR{ch_num}:VOLT:LOW {ch["low_v"]}')
                if wf == "PULS" and "width_s" in ch:
                    self.safe_write(f':SOUR{ch_num}:PULS:WIDT {ch["width_s"]}')

    def save_state_json(self, folder_path: str, step: int) -> bool:
        try:
            settings_folder = os.path.join(folder_path, "GeneratorSettings")
            os.makedirs(settings_folder, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            settings: Dict[str, Any] = {
                "device": "generator",
                "step": step,
                "timestamp": timestamp,
                "channel1": {},
                "channel2": {}
            }

            for ch_num, ch_key in [(1, "channel1"), (2, "channel2")]:
                wf = self.safe_query(f':SOUR{ch_num}:FUNC?')
                settings[ch_key]["waveform"] = wf
                if wf == "DC":
                    settings[ch_key]["offset_v"] = float(self.safe_query(f':SOUR{ch_num}:VOLT:OFFS?'))
                else:
                    settings[ch_key]["frequency_hz"] = float(self.safe_query(f':SOUR{ch_num}:FREQ?'))
                    settings[ch_key]["high_v"] = float(self.safe_query(f':SOUR{ch_num}:VOLT:HIGH?'))
                    settings[ch_key]["low_v"] = float(self.safe_query(f':SOUR{ch_num}:VOLT:LOW?'))
                    if wf == "PULS":
                        settings[ch_key]["width_s"] = float(self.safe_query(f':SOUR{ch_num}:PULS:WIDT?'))

            json_file = os.path.join(settings_folder, f"generator_step_{step:03d}.json")
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Ошибка сохранения состояния генератора: {e}")
            return False

    def close(self):
        try:
            self.dev.close()
        except Exception:
            pass


# ==============================================================================
# АППАРАТНЫЙ ДРАЙВЕР ОСЦИЛЛОГРАФА
# ==============================================================================
class RigolOscilloscopeDriver:
    def __init__(self, resource):
        self.dev = resource
        self.dev.timeout = 10000

    def safe_write(self, cmd: str):
        try:
            self.dev.write(cmd)
            time.sleep(0.05)
        except Exception as e:
            raise RuntimeError(f"Сбой записи в осциллограф [{cmd}]: {str(e)}")

    def safe_query(self, cmd: str) -> str:
        try:
            res = self.dev.query(cmd)
            time.sleep(0.05)
            return res.strip()
        except Exception as e:
            raise RuntimeError(f"Сбой запроса к осциллографу [{cmd}]: {str(e)}")

    def apply_state(self, state: OscilloscopeState):
        self.safe_write(f':CHAN1:SCAL {state.vscale}')
        self.safe_write(f':CHAN2:SCAL {state.vscale}')
        self.safe_write(f':TIM:SCAL {state.timebase}')
        self.safe_write(f':TRIG:EDGE:LEV {state.trigger_level}')

        ch1_disp = "ON" if state.ch1_enabled else "OFF"
        self.safe_write(f':CHAN1:DISP {ch1_disp}')
        if state.ch1_enabled:
            self.safe_write(f':CHAN1:OFFS {state.ch1_offset}')

        ch2_disp = "ON" if state.ch2_enabled else "OFF"
        self.safe_write(f':CHAN2:DISP {ch2_disp}')
        if state.ch2_enabled:
            self.safe_write(f':CHAN2:OFFS {state.ch2_offset}')

    def autoscale(self):
        self.safe_write(':AUT')
        time.sleep(2.0)

    def run(self):
        self.safe_write(':RUN')

    def stop(self):
        self.safe_write(':STOP')

    def save_state_json(self, folder_path: str, step: int):
        if step == -1:
            settings_folder = os.path.join(folder_path, "OscilloscopeSettings")
            filename = f"osc_settings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        else:
            settings_folder = os.path.join(folder_path, "OscilloscopeSettings", "ByStep")
            filename = f"osc_state_step_{step:03d}.json"

        os.makedirs(settings_folder, exist_ok=True)

        try: vscale1 = float(self.safe_query(':CHAN1:SCAL?'))
        except Exception: vscale1 = 1.0
        try: vscale2 = float(self.safe_query(':CHAN2:SCAL?'))
        except Exception: vscale2 = 1.0
        try: timebase = float(self.safe_query(':TIM:SCAL?'))
        except Exception: timebase = 1e-6
        try: ch1_enabled = (self.safe_query(':CHAN1:DISP?') == '1')
        except Exception: ch1_enabled = True
        try: ch2_enabled = (self.safe_query(':CHAN2:DISP?') == '1')
        except Exception: ch2_enabled = True
        try: ch1_offset = float(self.safe_query(':CHAN1:OFFS?'))
        except Exception: ch1_offset = 0.0
        try: ch2_offset = float(self.safe_query(':CHAN2:OFFS?'))
        except Exception: ch2_offset = 0.0

        try: ch1_imp = self.safe_query(':CHAN1:IMP?')
        except Exception: ch1_imp = "UNKNOWN"
        try: ch2_imp = self.safe_query(':CHAN2:IMP?')
        except Exception: ch2_imp = "UNKNOWN"
        try: trig_lev = float(self.safe_query(':TRIG:EDGE:LEV?'))
        except Exception: trig_lev = 0.0
        try: trig_swe = self.safe_query(':TRIG:SWE?')
        except Exception: trig_swe = "UNKNOWN"
        try: trig_mod = self.safe_query(':TRIG:MODE?')
        except Exception: trig_mod = "UNKNOWN"
        try: trig_src = self.safe_query(':TRIG:EDGE:SOUR?')
        except Exception: trig_src = "UNKNOWN"
        try: trig_slp = self.safe_query(':TRIG:EDGE:SLOP?')
        except Exception: trig_slp = "UNKNOWN"

        settings = {
            "step": step if step >= 0 else None,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "channel1": {
                "enabled": ch1_enabled,
                "scale_vdiv": vscale1,
                "offset_v": ch1_offset,
                "impedance": ch1_imp
            },
            "channel2": {
                "enabled": ch2_enabled,
                "scale_vdiv": vscale2,
                "offset_v": ch2_offset,
                "impedance": ch2_imp
            },
            "timebase_sdiv": timebase,
            "trigger": {
                "level_v": trig_lev,
                "sweep_mode": trig_swe,
                "type": trig_mod,
                "source": trig_src,
                "slope": trig_slp
            }
        }

        json_file = os.path.join(settings_folder, filename)
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=4, ensure_ascii=False)

    def load_state_from_json(self, json_file: str):
        with open(json_file, 'r', encoding='utf-8') as f:
            settings = json.load(f)

        if "channel1" in settings:
            ch1 = settings["channel1"]
            if "scale_vdiv" in ch1: self.safe_write(f':CHAN1:SCAL {ch1["scale_vdiv"]}')
            if "enabled" in ch1: self.safe_write(f':CHAN1:DISP {"ON" if ch1["enabled"] else "OFF"}')
            if "offset_v" in ch1: self.safe_write(f':CHAN1:OFFS {ch1["offset_v"]}')

        if "channel2" in settings:
            ch2 = settings["channel2"]
            if "scale_vdiv" in ch2: self.safe_write(f':CHAN2:SCAL {ch2["scale_vdiv"]}')
            if "enabled" in ch2: self.safe_write(f':CHAN2:DISP {"ON" if ch2["enabled"] else "OFF"}')
            if "offset_v" in ch2: self.safe_write(f':CHAN2:OFFS {ch2["offset_v"]}')

        if "timebase_sdiv" in settings:
            self.safe_write(f':TIM:SCAL {settings["timebase_sdiv"]}')

        if "trigger" in settings:
            trig = settings["trigger"]
            if "level_v" in trig: self.safe_write(f':TRIG:EDGE:LEV {trig["level_v"]}')
            if "sweep_mode" in trig: self.safe_write(f':TRIG:SWE {trig["sweep_mode"]}')

    def get_screenshot(self, filepath: str):
        try:
            self.safe_write(':SAVE:IMAGE:FORMAT PNG')
            time.sleep(0.4)
            self.safe_write(':DISP:DATA?')
            raw_data = self.dev.read_raw()
            if raw_data.startswith(b'#'):
                n_len = int(chr(raw_data[1]))
                img_data = raw_data[2 + n_len:]
            else:
                img_data = raw_data

            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                f.write(img_data)
        except Exception as e:
            raise RuntimeError(f"Ошибка сохранения скриншота осциллографа: {str(e)}")

    def save_channel_data(self, channel: int, step: int, voltage: float,
                          waveforms_folder: str, measurements_folder: str,
                          screenshots_folder: Optional[str] = None) -> Dict[str, Any]:
        measurement_time = datetime.now().strftime("%H:%M:%S")
        voltage_str = f"{voltage:.2f}".replace('.', '_')
        base_name = f"CH{channel}_{voltage_str}V_step{step:03d}"

        self.safe_write(f':CHAN{channel}:DISP ON')
        self.run()
        time.sleep(0.4)
        self.stop()
        time.sleep(0.2)

        self.safe_write(f':WAV:SOUR CHAN{channel}')
        self.safe_write(':WAV:MODE RAW')
        self.safe_write(':WAV:FORM BYTE')

        preamble = self.safe_query(':WAV:PREAMBLE?').split(',')
        points = int(preamble[2])
        x_inc = float(preamble[4])
        x_orig = float(preamble[5])
        y_inc = float(preamble[7])
        y_orig = float(preamble[8])
        y_ref = float(preamble[9])

        self.safe_write(":WAV:STAR 1")
        self.safe_write(f":WAV:STOP {points}")

        old_timeout = self.dev.timeout
        self.dev.timeout = 15000
        try:
            self.dev.write(':WAV:DATA?')
            raw_data = self.dev.read_raw()
        finally:
            self.dev.timeout = old_timeout

        header_len = int(chr(raw_data[1]))
        adc_data = list(raw_data[2 + header_len: -1])
        if not adc_data:
            raise RuntimeError(f"Пустые данные выборки канала {channel}")

        times = [x_orig + (n * x_inc) for n in range(len(adc_data))]
        volts = [(val - y_ref) * y_inc + y_orig for val in adc_data]

        self.safe_write(f':MEASURE:SOURCE CHAN{channel}')
        try: vrms = float(self.safe_query(':MEASURE:VRMS?'))
        except Exception: vrms = 0.0
        try: freq = float(self.safe_query(':MEASURE:FREQ?'))
        except Exception: freq = 0.0
        try: period = float(self.safe_query(':MEASURE:PERIOD?'))
        except Exception: period = 0.0
        try: pwidth = float(self.safe_query(':MEASURE:PWIDTH?'))
        except Exception: pwidth = 0.0

        os.makedirs(waveforms_folder, exist_ok=True)
        wave_file = os.path.join(waveforms_folder, f"{base_name}_wave.csv")
        pd.DataFrame({'Time_s': times, 'Voltage_V': volts}).to_csv(
            wave_file, index=False, sep=';', decimal=','
        )

        os.makedirs(measurements_folder, exist_ok=True)
        measurements = {
            'step': step, 'voltage': voltage, 'vrms': vrms,
            'frequency': freq, 'period': period, 'pulse_width': pwidth,
            'time': measurement_time
        }

        for metric in ['vrms', 'frequency', 'period', 'pulse_width']:
            f_path = os.path.join(measurements_folder, f"{metric}_CH{channel}.csv")
            row = {'step': step, 'voltage': voltage, metric: measurements[metric], 'time': measurement_time}
            df = pd.DataFrame([row])
            mode = 'a' if os.path.exists(f_path) else 'w'
            df.to_csv(f_path, mode=mode, header=not os.path.exists(f_path), index=False, sep=';', decimal=',')

        if screenshots_folder:
            screenshot_path = os.path.join(screenshots_folder, f"{base_name}_screenshot.png")
            try:
                self.get_screenshot(screenshot_path)
            except Exception as e:
                print(f"Ошибка сохранения скриншота: {e}")

        return measurements

    def close(self):
        try:
            self.dev.close()
        except Exception:
            pass


# ==============================================================================
# АППАРАТНЫЙ ДРАЙВЕР СПЕКТРОАНАЛИЗАТОРА (DSA)
# ==============================================================================
class SpectrumAnalyzerDriver:
    def __init__(self, resource):
        self.dev = resource
        self.dev.timeout = 10000

    def safe_write(self, cmd: str):
        try:
            self.dev.write(cmd)
            time.sleep(0.1)
        except Exception as e:
            raise RuntimeError(f"Сбой записи в DSA [{cmd}]: {str(e)}")

    def safe_query(self, cmd: str) -> str:
        try:
            res = self.dev.query(cmd)
            time.sleep(0.1)
            return res.strip()
        except Exception as e:
            raise RuntimeError(f"Сбой запроса к DSA [{cmd}]: {str(e)}")

    def apply_settings(self, center_freq: float, span: float, rbw: float, vbw: Optional[float] = None):
        if vbw is None:
            vbw = rbw
        self.safe_write(f':SENS:FREQ:CENT {center_freq}')
        self.safe_write(f':SENS:FREQ:SPAN {span}')
        self.safe_write(f':SENS:BAND:RES {rbw}')
        self.safe_write(f':SENS:BAND:VIDEO {vbw}')

    def find_peak(self) -> Tuple[float, float]:
        self.safe_write(':CALC:MARK1:STAT ON')
        self.safe_write(':CALC:MARK1:MAX')
        time.sleep(0.3)
        try: peak_freq = float(self.safe_query(':CALC:MARK1:X?'))
        except Exception: peak_freq = 0.0
        try: peak_amp = float(self.safe_query(':CALC:MARK1:Y?'))
        except Exception: peak_amp = 0.0
        return peak_freq, peak_amp

    def save_state_json(self, folder_path: str, step: int, mode_name: str = "manual"):
        if step == -1:
            settings_folder = os.path.join(folder_path, "AnalyzerSettings")
            filename = f"sa_settings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        else:
            settings_folder = os.path.join(folder_path, "AnalyzerSettings", "ByStep")
            filename = f"sa_state_step_{step:03d}.json"

        os.makedirs(settings_folder, exist_ok=True)
        settings: Dict[str, Any] = {
            "device": "spectrum_analyzer",
            "step": step if step >= 0 else None,
            "mode": mode_name,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
        }

        try: settings["center_freq_hz"] = float(self.safe_query(':SENS:FREQ:CENT?'))
        except Exception: settings["center_freq_hz"] = 0.0
        try: settings["span_hz"] = float(self.safe_query(':SENS:FREQ:SPAN?'))
        except Exception: settings["span_hz"] = 0.0
        try: settings["rbw_hz"] = float(self.safe_query(':SENS:BAND:RES?'))
        except Exception: settings["rbw_hz"] = 0.0
        try: settings["vbw_hz"] = float(self.safe_query(':SENS:BAND:VIDEO?'))
        except Exception: settings["vbw_hz"] = 0.0

        json_file = os.path.join(settings_folder, filename)
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=4, ensure_ascii=False)

    def load_state_from_json(self, json_file: str):
        with open(json_file, 'r', encoding='utf-8') as f:
            settings = json.load(f)

        if "center_freq_hz" in settings:
            self.safe_write(f':SENS:FREQ:CENT {settings["center_freq_hz"]}')
        if "span_hz" in settings:
            self.safe_write(f':SENS:FREQ:SPAN {settings["span_hz"]}')
        if "rbw_hz" in settings:
            self.safe_write(f':SENS:BAND:RES {settings["rbw_hz"]}')
        if "vbw_hz" in settings:
            self.safe_write(f':SENS:BAND:VIDEO {settings["vbw_hz"]}')

    def save_trace_csv(self, traces_folder: str, step: int, voltage: float, gen_freq: float = 0):
        try:
            self.safe_write(':INIT:CONT OFF')
            self.safe_write(':INIT:IMM')
            time.sleep(2.0)

            self.safe_write(':FORM ASCII')
            raw_response = self.safe_query(':TRACE:DATA? TRACE1')

            if raw_response.startswith('#'):
                comma_pos = raw_response.find(',')
                if comma_pos > 0:
                    data_part = raw_response[comma_pos + 1:]
                else:
                    raise RuntimeError("Разделитель данных не найден в ответе трассы.")
            else:
                data_part = raw_response

            amps = []
            for v in data_part.split(','):
                v = v.strip()
                if v:
                    try: amps.append(float(v))
                    except ValueError: continue

            if not amps:
                raise RuntimeError("Пустая трасса спектроанализатора.")

            amps_array = np.array(amps)
            num_points = len(amps_array)

            start_freq_hz = float(self.safe_query(':SENS:FREQ:START?'))
            stop_freq_hz = float(self.safe_query(':SENS:FREQ:STOP?'))
            freqs_mhz = np.linspace(start_freq_hz / 1e6, stop_freq_hz / 1e6, num_points)

            os.makedirs(traces_folder, exist_ok=True)
            voltage_str = f"{voltage:.2f}".replace('.', '_')
            if gen_freq > 0:
                filename = f"SA_{gen_freq:.0f}Hz_{voltage_str}V_step{step:03d}.csv"
            else:
                filename = f"SA_step{step:03d}_{voltage_str}V.csv"

            filepath = os.path.join(traces_folder, filename)
            df = pd.DataFrame({'Frequency_MHz': freqs_mhz, 'Amplitude_dBm': amps_array})
            df.to_csv(filepath, index=False, sep=';', decimal=',', encoding='utf-8-sig')

            self.safe_write(':INIT:CONT ON')
            return freqs_mhz, amps_array
        except Exception as e:
            try: self.safe_write(':INIT:CONT ON')
            except Exception: pass
            raise RuntimeError(f"Сбой сохранения трассы анализатора: {str(e)}")

    def close(self):
        try:
            self.dev.close()
        except Exception:
            pass


# ==============================================================================
# АППАРАТНЫЙ ДРАЙВЕР ИЗМЕРИТЕЛЯ МОЩНОСТИ (PM100USB)
# ==============================================================================
class PowerMeterDriver:
    def __init__(self, resource):
        self.dev = resource
        try:
            self.dev.baud_rate = 115200
            self.dev.read_termination = '\r\n'
            self.dev.write_termination = '\r\n'
            self.dev.timeout = 2000
        except Exception as e:
            raise RuntimeError(f"Сбой конфигурации порта PM100USB: {str(e)}")

    def measure_power(self) -> Dict[str, Any]:
        try:
            time.sleep(0.3)
            self.dev.write("*POW_S?")
            time.sleep(0.1)
            resp = self.dev.read().strip()

            power_w = 0.0
            power_dbm = -200.0
            raw_unit = "None"

            if 'LOW' in resp:
                raw_unit = 'LOW'
            elif 'HIGH' in resp:
                raw_unit = 'HIGH'
                power_w = 999.0
                power_dbm = 999.0
            else:
                match = re.search(r'([-+]?\d*\.\d+|\d+)', resp)
                if match:
                    val = float(match.group(1))
                    if 'dBm' in resp:
                        raw_unit = 'dBm'
                        power_dbm = val
                        power_w = (10 ** (val / 10.0)) / 1000.0
                    elif 'mW' in resp:
                        raw_unit = 'mW'
                        power_w = val / 1000.0
                        power_dbm = 10.0 * math.log10(val) if val > 0 else -200.0
                    elif 'W' in resp:
                        raw_unit = 'W'
                        power_w = val
                        power_dbm = 10.0 * math.log10(val * 1000.0) if val > 0 else -200.0
                else:
                    raise RuntimeError(f"Неизвестный ответ PM100USB: {resp}")

            return {
                'raw_unit': raw_unit,
                'power_w': power_w,
                'power_dbm': power_dbm
            }
        except Exception as e:
            raise RuntimeError(f"Ошибка чтения PM100USB: {str(e)}")

    def close(self):
        try:
            self.dev.close()
        except Exception:
            pass


# ==============================================================================
# АППАРАТНЫЙ ДРАЙВЕР ОПТИЧЕСКОГО АНАЛИЗАТОРА (LSA)
# ==============================================================================
class LSADriver:
    def __init__(self, host: str = "127.0.0.1", port: int = 5006, timeout: float = 2.0):
        self.host = host
        self.port = port
        self.timeout = timeout

    def save_smx_and_get_peak(self, smx_path: str) -> float:
        abs_smx_path = os.path.abspath(smx_path)
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.settimeout(self.timeout)
                cmd = {"action": "save", "path": abs_smx_path}
                sock.sendto(json.dumps(cmd).encode('utf-8'), (self.host, self.port))
                resp_data, _ = sock.recvfrom(4096)
                resp = json.loads(resp_data.decode('utf-8'))

            if not os.path.exists(abs_smx_path):
                err = resp.get('error', 'Файл SMX не создан')
                raise RuntimeError(f"Мост вернул ошибку: {err}")

            return float(resp.get("peak", 0.0))
        except socket.timeout:
            raise RuntimeError("Таймаут: 32-битный мост LSA (порт 5006) не отвечает.")
        except Exception as e:
            raise RuntimeError(f"Сбой связи с мостом LSA: {str(e)}")

    def convert_smx_to_csv(self, smx_path: str, csv_path: str) -> str:
        try:
            with open(smx_path, "r", encoding="latin-1") as f:
                lines = [l.strip() for l in f]

            def get_tag_value(tag: str) -> Optional[float]:
                if tag in lines:
                    idx = lines.index(tag)
                    return float(lines[idx + 2].replace(',', '.'))
                return None

            x_start = get_tag_value("100062")
            x_step = get_tag_value("100061")

            if "10003" not in lines:
                raise RuntimeError(f"Тег '10003' не найден в {os.path.basename(smx_path)}")

            idx_10003 = lines.index("10003")
            raw_vals = []
            for line in lines[idx_10003 + 2:]:
                if line.startswith("10004") or " " in line:
                    break
                try:
                    raw_vals.append(float(line.replace(",", ".")))
                except ValueError:
                    break

            y_raw = np.array(raw_vals)
            num_points = len(y_raw)
            if num_points == 0:
                raise RuntimeError("Пустой спектр интенсивности в SMX.")

            if x_start is None: x_start = 1614.7115
            if x_step is None: x_step = -0.0630067

            x_raw = x_start + np.arange(num_points) * x_step
            x = x_raw[::-1]
            y = y_raw[::-1]

            df = pd.DataFrame({"Wavelength_nm": x, "Intensity": y})
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            df.to_csv(csv_path, index=False, sep=";", decimal=',')
            return csv_path
        except Exception as e:
            raise RuntimeError(f"Ошибка конвертации SMX->CSV: {str(e)}")


# ==============================================================================
# UI ПАНЕЛИ
# ==============================================================================
class ChannelPanel(QGroupBox):
    output_toggled_signal = pyqtSignal(int, bool)

    def __init__(self, channel_num: int):
        super().__init__(f"Канал {channel_num}")
        self.ch_num = channel_num
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        self.btn_output = QPushButton("Выход ВЫКЛ")
        self.btn_output.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
        self.btn_output.setCheckable(True)
        self.btn_output.toggled.connect(self.on_output_toggled)
        layout.addWidget(self.btn_output)

        wave_layout = QHBoxLayout()
        wave_layout.addWidget(QLabel("Форма:"))
        self.combo_wave = QComboBox()
        self.combo_wave.addItems(["SIN", "PULS", "DC"])
        self.combo_wave.currentTextChanged.connect(self.update_visibility)
        wave_layout.addWidget(self.combo_wave)
        layout.addLayout(wave_layout)

        self.w_freq, self.freq_val = self.create_param_row("Частота (Гц):", "1000000", "100000", layout)
        self.w_high, self.high_val = self.create_param_row("High Level (В):", "1.0", "0.1", layout)
        self.w_low, self.low_val = self.create_param_row("Low Level (В):", "-1.0", "0.1", layout)
        self.w_width, self.width_val = self.create_param_row("Длительность (с):", "0.000001", "0.0000001", layout)
        self.w_offset, self.offset_val = self.create_param_row("Смещение (В):", "0.0", "0.1", layout)

        self.update_visibility(self.combo_wave.currentText())
        layout.addStretch()

    def create_param_row(self, label: str, default_val: str, default_step: str, parent_layout):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel(label))
        val_input = QLineEdit(default_val)
        val_input.setFixedWidth(90)
        layout.addWidget(val_input)

        layout.addWidget(QLabel("Шаг:"))
        step_input = QLineEdit(default_step)
        step_input.setFixedWidth(70)
        layout.addWidget(step_input)

        btn_up = QPushButton("+")
        btn_down = QPushButton("-")
        btn_up.setFixedWidth(30)
        btn_down.setFixedWidth(30)

        btn_up.clicked.connect(lambda: self.adjust_value(val_input, step_input, 1))
        btn_down.clicked.connect(lambda: self.adjust_value(val_input, step_input, -1))

        layout.addWidget(btn_up)
        layout.addWidget(btn_down)
        parent_layout.addWidget(widget)
        return widget, val_input

    def adjust_value(self, val_widget, step_widget, direction):
        try:
            val = float(val_widget.text())
            step = float(step_widget.text())
            new_val = val + (step * direction)
            val_widget.setText(f"{new_val:.6f}".rstrip('0').rstrip('.'))
        except ValueError:
            pass

    def on_output_toggled(self, checked):
        if checked:
            self.btn_output.setText("Выход ВКЛ")
            self.btn_output.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        else:
            self.btn_output.setText("Выход ВЫКЛ")
            self.btn_output.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
        self.output_toggled_signal.emit(self.ch_num, checked)

    def update_visibility(self, waveform: str):
        is_dc = (waveform == "DC")
        is_pulse = (waveform == "PULS")
        self.w_freq.setVisible(not is_dc)
        self.w_high.setVisible(not is_dc)
        self.w_low.setVisible(not is_dc)
        self.w_width.setVisible(is_pulse)
        self.w_offset.setVisible(is_dc)

    def get_state(self) -> ChannelState:
        return ChannelState(
            enabled=self.btn_output.isChecked(),
            waveform=self.combo_wave.currentText(),
            freq=float(self.freq_val.text() or 0),
            high=float(self.high_val.text() or 0),
            low=float(self.low_val.text() or 0),
            width=float(self.width_val.text() or 0),
            offset=float(self.offset_val.text() or 0)
        )


class OscilloscopePanel(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        mode_box = QHBoxLayout()
        mode_box.addWidget(QLabel("Режим автонастройки:"))
        self.combo_auto_mode = QComboBox()
        self.combo_auto_mode.addItems(["Вручную (без изменений)", "Автонастройка на каждом шаге"])
        mode_box.addWidget(self.combo_auto_mode)
        mode_box.addStretch()
        layout.addLayout(mode_box)

        channels_layout = QHBoxLayout()
        self.ch1_check = QCheckBox("Канал 1 ВКЛ")
        self.ch1_check.setChecked(True)
        self.ch2_check = QCheckBox("Канал 2 ВКЛ")
        self.ch2_check.setChecked(True)
        channels_layout.addWidget(self.ch1_check)
        channels_layout.addWidget(self.ch2_check)
        channels_layout.addStretch()
        layout.addLayout(channels_layout)

        self.w_ch1_off, self.ch1_offset = self.create_param_row("Смещение CH1 (В):", "0.0", "0.1")
        self.w_ch2_off, self.ch2_offset = self.create_param_row("Смещение CH2 (В):", "0.0", "0.1")
        self.w_vscale, self.vscale = self.create_param_row("Масштаб Y (В/дел):", "1.0", "0.1")
        self.w_timebase, self.timebase = self.create_param_row("Развертка X (с/дел):", "0.0000001", "0.00000005")
        self.w_trigger, self.trigger = self.create_param_row("Уровень триггера (В):", "0.5", "0.1")

        layout.addWidget(self.w_ch1_off)
        layout.addWidget(self.w_ch2_off)
        layout.addWidget(self.w_vscale)
        layout.addWidget(self.w_timebase)
        layout.addWidget(self.w_trigger)
        layout.addStretch()

    def create_param_row(self, label: str, default_val: str, default_step: str):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel(label))
        val_input = QLineEdit(default_val)
        val_input.setFixedWidth(90)
        layout.addWidget(val_input)

        layout.addWidget(QLabel("Шаг:"))
        step_input = QLineEdit(default_step)
        step_input.setFixedWidth(70)
        layout.addWidget(step_input)

        btn_up = QPushButton("+")
        btn_down = QPushButton("-")
        btn_up.setFixedWidth(30)
        btn_down.setFixedWidth(30)

        btn_up.clicked.connect(lambda: self.adjust_value(val_input, step_input, 1))
        btn_down.clicked.connect(lambda: self.adjust_value(val_input, step_input, -1))

        layout.addWidget(btn_up)
        layout.addWidget(btn_down)
        return widget, val_input

    def adjust_value(self, val_widget, step_widget, direction):
        try:
            val = float(val_widget.text())
            step = float(step_widget.text())
            new_val = val + (step * direction)
            val_widget.setText(f"{new_val:.6f}".rstrip('0').rstrip('.'))
        except ValueError:
            pass

    def is_auto_scale_enabled(self) -> bool:
        return self.combo_auto_mode.currentIndex() == 1

    def get_state(self) -> OscilloscopeState:
        return OscilloscopeState(
            ch1_enabled=self.ch1_check.isChecked(),
            ch2_enabled=self.ch2_check.isChecked(),
            ch1_offset=float(self.ch1_offset.text() or 0),
            ch2_offset=float(self.ch2_offset.text() or 0),
            vscale=float(self.vscale.text() or 1.0),
            timebase=float(self.timebase.text() or 1e-7),
            trigger_level=float(self.trigger.text() or 0.5)
        )


class SpectrumAnalyzerPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        mode_box = QHBoxLayout()
        mode_box.addWidget(QLabel("Режим автонастройки:"))
        self.combo_auto_mode = QComboBox()
        self.combo_auto_mode.addItems(["Вручную (без изменений)", "Автоматически (под частоту генератора)"])
        mode_box.addWidget(self.combo_auto_mode)
        mode_box.addStretch()
        layout.addLayout(mode_box)

        self.w_center, self.center_freq = self.create_param_row("Центр. частота (Гц):", "1000000", "100000")
        self.w_span, self.span = self.create_param_row("SPAN (Гц):", "200000", "50000")
        self.w_rbw, self.rbw = self.create_param_row("RBW / VBW (Гц):", "10000", "1000")

        layout.addWidget(self.w_center)
        layout.addWidget(self.w_span)
        layout.addWidget(self.w_rbw)
        layout.addStretch()

    def create_param_row(self, label: str, default_val: str, default_step: str):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel(label))
        val_input = QLineEdit(default_val)
        val_input.setFixedWidth(90)
        layout.addWidget(val_input)

        layout.addWidget(QLabel("Шаг:"))
        step_input = QLineEdit(default_step)
        step_input.setFixedWidth(70)
        layout.addWidget(step_input)

        btn_up = QPushButton("+")
        btn_down = QPushButton("-")
        btn_up.setFixedWidth(30)
        btn_down.setFixedWidth(30)

        btn_up.clicked.connect(lambda: self.adjust_value(val_input, step_input, 1))
        btn_down.clicked.connect(lambda: self.adjust_value(val_input, step_input, -1))

        layout.addWidget(btn_up)
        layout.addWidget(btn_down)
        return widget, val_input

    def adjust_value(self, val_widget, step_widget, direction):
        try:
            val = float(val_widget.text())
            step = float(step_widget.text())
            new_val = val + (step * direction)
            val_widget.setText(f"{new_val:.6f}".rstrip('0').rstrip('.'))
        except ValueError:
            pass

    def is_auto_tune_enabled(self) -> bool:
        return self.combo_auto_mode.currentIndex() == 1

    def get_state(self) -> SpectrumAnalyzerState:
        return SpectrumAnalyzerState(
            center_freq=float(self.center_freq.text() or 1e6),
            span=float(self.span.text() or 200000),
            rbw=float(self.rbw.text() or 10000),
            vbw=float(self.rbw.text() or 10000)
        )


class OpticsPowerPanel(QWidget):
    manual_power_requested = pyqtSignal()
    convert_smx_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        pm_group = QGroupBox("Измеритель мощности (PM100USB)")
        pm_layout = QVBoxLayout(pm_group)
        self.btn_meas_power = QPushButton("Сделать разовый замер мощности")
        self.btn_meas_power.clicked.connect(self.manual_power_requested.emit)
        self.lbl_power_result = QLabel("Результат: Нет данных")
        self.lbl_power_result.setStyleSheet("font-weight: bold;")
        pm_layout.addWidget(self.btn_meas_power)
        pm_layout.addWidget(self.lbl_power_result)
        layout.addWidget(pm_group)

        lsa_group = QGroupBox("Оптический анализатор (LSA)")
        lsa_layout = QVBoxLayout(lsa_group)
        self.btn_convert_smx = QPushButton("Выбрать и конвертировать .smx в .csv")
        self.btn_convert_smx.clicked.connect(self.convert_smx_requested.emit)
        lsa_layout.addWidget(self.btn_convert_smx)
        layout.addWidget(lsa_group)

        layout.addStretch()

    def update_power_result(self, res: Optional[Dict[str, Any]]):
        if res:
            w = res.get('power_w', 0.0)
            dbm = res.get('power_dbm', 0.0)
            self.lbl_power_result.setText(f"Результат: {w * 1000.0:.3f} мВт ({dbm:.2f} дБм)")
        else:
            self.lbl_power_result.setText("Результат: Ошибка замера")


# ==============================================================================
# ПАНЕЛЬ И ЗАДАЧА ЭКСПЕРИМЕНТА
# ==============================================================================
@dataclass
class ExperimentTask:
    save_dir: str
    mode: str
    values: list
    pause_s: float
    target_ch: str

    use_gen: bool = True
    use_osc: bool = True
    use_sa: bool = True
    use_pm: bool = True
    use_lsa: bool = True

    osc_ch1_enabled: bool = True
    osc_ch2_enabled: bool = True
    osc_auto: bool = False
    sa_auto: bool = False
    sa_gen_freq: float = 1e6


class ExperimentPanel(QWidget):
    start_requested = pyqtSignal(ExperimentTask)
    stop_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        dev_group = QGroupBox("Приборы в цикле")
        dev_layout = QHBoxLayout(dev_group)
        self.check_gen = QCheckBox("Генератор")
        self.check_gen.setChecked(True)
        self.check_osc = QCheckBox("Осциллограф")
        self.check_osc.setChecked(True)
        self.check_sa = QCheckBox("Спектроанализатор")
        self.check_sa.setChecked(True)
        self.check_pm = QCheckBox("Измеритель мощности")
        self.check_pm.setChecked(True)
        self.check_lsa = QCheckBox("LSA (SMX)")
        self.check_lsa.setChecked(True)

        dev_layout.addWidget(self.check_gen)
        dev_layout.addWidget(self.check_osc)
        dev_layout.addWidget(self.check_sa)
        dev_layout.addWidget(self.check_pm)
        dev_layout.addWidget(self.check_lsa)
        layout.addWidget(dev_group)

        folder_group = QGroupBox("Каталог сохранения")
        f_layout = QHBoxLayout(folder_group)
        self.path_input = QLineEdit(str(Path.home() / 'Rigol_Measurements'))
        self.btn_browse = QPushButton("Обзор...")
        self.btn_browse.clicked.connect(self.browse_folder)
        f_layout.addWidget(self.path_input)
        f_layout.addWidget(self.btn_browse)
        layout.addWidget(folder_group)

        mode_group = QGroupBox("Параметры развёртки")
        m_layout = QVBoxLayout(mode_group)

        top_m = QHBoxLayout()
        top_m.addWidget(QLabel("Режим:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["DC", "SIN параметры", "PULSE параметры"])
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        top_m.addWidget(self.mode_combo)

        top_m.addWidget(QLabel("Целевой канал генератора:"))
        self.target_ch_combo = QComboBox()
        self.target_ch_combo.addItems(["Канал 1", "Канал 2", "Оба канала"])
        top_m.addWidget(self.target_ch_combo)

        top_m.addWidget(QLabel("Пауза (с):"))
        self.pause_input = QLineEdit("2.0")
        self.pause_input.setFixedWidth(50)
        top_m.addWidget(self.pause_input)
        m_layout.addLayout(top_m)

        self.dc_box = QWidget()
        dc_l = QHBoxLayout(self.dc_box)
        dc_l.addWidget(QLabel("Старт (В):"))
        self.dc_start = QLineEdit("0.0")
        dc_l.addWidget(self.dc_start)
        dc_l.addWidget(QLabel("Шаг (В):"))
        self.dc_step = QLineEdit("0.5")
        dc_l.addWidget(self.dc_step)
        dc_l.addWidget(QLabel("Кол-во шагов:"))
        self.dc_steps = QLineEdit("5")
        dc_l.addWidget(self.dc_steps)
        m_layout.addWidget(self.dc_box)

        self.sin_box = QWidget()
        sin_l = QGridLayout(self.sin_box)
        sin_l.addWidget(QLabel("Старт частота (Гц):"), 0, 0)
        self.sin_f_start = QLineEdit("1000000")
        sin_l.addWidget(self.sin_f_start, 0, 1)
        sin_l.addWidget(QLabel("Шаг частоты:"), 0, 2)
        self.sin_f_step = QLineEdit("100000")
        sin_l.addWidget(self.sin_f_step, 0, 3)

        sin_l.addWidget(QLabel("High (В):"), 1, 0)
        self.sin_h_start = QLineEdit("1.0")
        sin_l.addWidget(self.sin_h_start, 1, 1)
        sin_l.addWidget(QLabel("Шаг High:"), 1, 2)
        self.sin_h_step = QLineEdit("0.1")
        sin_l.addWidget(self.sin_h_step, 1, 3)

        sin_l.addWidget(QLabel("Low (В):"), 2, 0)
        self.sin_l_start = QLineEdit("-1.0")
        sin_l.addWidget(self.sin_l_start, 2, 1)
        sin_l.addWidget(QLabel("Шаг Low:"), 2, 2)
        self.sin_l_step = QLineEdit("0.1")
        sin_l.addWidget(self.sin_l_step, 2, 3)

        sin_l.addWidget(QLabel("Шагов:"), 3, 0)
        self.sin_steps = QLineEdit("5")
        sin_l.addWidget(self.sin_steps, 3, 1)
        m_layout.addWidget(self.sin_box)
        self.sin_box.setVisible(False)

        self.pulse_box = QWidget()
        pulse_l = QGridLayout(self.pulse_box)
        pulse_l.addWidget(QLabel("Старт частота (Гц):"), 0, 0)
        self.pulse_f_start = QLineEdit("1000000")
        pulse_l.addWidget(self.pulse_f_start, 0, 1)
        pulse_l.addWidget(QLabel("Шаг частоты:"), 0, 2)
        self.pulse_f_step = QLineEdit("100000")
        pulse_l.addWidget(self.pulse_f_step, 0, 3)

        pulse_l.addWidget(QLabel("High (В):"), 1, 0)
        self.pulse_h_start = QLineEdit("1.0")
        pulse_l.addWidget(self.pulse_h_start, 1, 1)
        pulse_l.addWidget(QLabel("Шаг High:"), 1, 2)
        self.pulse_h_step = QLineEdit("0.1")
        pulse_l.addWidget(self.pulse_h_step, 1, 3)

        pulse_l.addWidget(QLabel("Low (В):"), 2, 0)
        self.pulse_l_start = QLineEdit("-1.0")
        pulse_l.addWidget(self.pulse_l_start, 2, 1)
        pulse_l.addWidget(QLabel("Шаг Low:"), 2, 2)
        self.pulse_l_step = QLineEdit("0.1")
        pulse_l.addWidget(self.pulse_l_step, 2, 3)

        pulse_l.addWidget(QLabel("Ширина (с):"), 3, 0)
        self.pulse_w_start = QLineEdit("0.000001")
        pulse_l.addWidget(self.pulse_w_start, 3, 1)
        pulse_l.addWidget(QLabel("Шаг ширины:"), 3, 2)
        self.pulse_w_step = QLineEdit("0.0000001")
        pulse_l.addWidget(self.pulse_w_step, 3, 3)

        pulse_l.addWidget(QLabel("Шагов:"), 4, 0)
        self.pulse_steps = QLineEdit("5")
        pulse_l.addWidget(self.pulse_steps, 4, 1)
        m_layout.addWidget(self.pulse_box)
        self.pulse_box.setVisible(False)

        layout.addWidget(mode_group)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        self.log_window = QTextEdit()
        self.log_window.setReadOnly(True)
        layout.addWidget(self.log_window)

        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("Запустить эксперимент")
        self.btn_start.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
        self.btn_start.clicked.connect(self.collect_task_and_emit)

        self.btn_stop = QPushButton("Остановить")
        self.btn_stop.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 8px;")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_requested.emit)

        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)
        layout.addLayout(btn_layout)

    def on_mode_changed(self, mode: str):
        self.dc_box.setVisible(mode == "DC")
        self.sin_box.setVisible(mode == "SIN параметры")
        self.pulse_box.setVisible(mode == "PULSE параметры")

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Каталог сохранения")
        if folder:
            self.path_input.setText(folder)

    def log(self, text: str):
        self.log_window.append(text)

    def collect_task_and_emit(self):
        folder = self.path_input.text().strip()
        if not folder:
            QMessageBox.warning(self, "Ошибка", "Укажите каталог для сохранения!")
            return

        mode = self.mode_combo.currentText()
        try:
            pause = float(self.pause_input.text())
            if mode == "DC":
                s = float(self.dc_start.text())
                step = float(self.dc_step.text())
                n = int(self.dc_steps.text())
                vals = [s + i * step for i in range(n)]
            elif mode == "SIN параметры":
                n = int(self.sin_steps.text())
                f_s, f_st = float(self.sin_f_start.text()), float(self.sin_f_step.text())
                h_s, h_st = float(self.sin_h_start.text()), float(self.sin_h_step.text())
                l_s, l_st = float(self.sin_l_start.text()), float(self.sin_l_step.text())
                vals = [(f_s + i * f_st, h_s + i * h_st, l_s + i * l_st) for i in range(n)]
            else:
                n = int(self.pulse_steps.text())
                f_s, f_st = float(self.pulse_f_start.text()), float(self.pulse_f_step.text())
                h_s, h_st = float(self.pulse_h_start.text()), float(self.pulse_h_step.text())
                l_s, l_st = float(self.pulse_l_start.text()), float(self.pulse_l_step.text())
                w_s, w_st = float(self.pulse_w_start.text()), float(self.pulse_w_step.text())
                vals = [(f_s + i * f_st, h_s + i * h_st, l_s + i * l_st, w_s + i * w_st) for i in range(n)]
        except ValueError:
            QMessageBox.warning(self, "Ошибка", "Проверьте корректность введённых числовых параметров!")
            return

        task = ExperimentTask(
            save_dir=folder,
            mode=mode,
            values=vals,
            pause_s=pause,
            target_ch=self.target_ch_combo.currentText(),
            use_gen=self.check_gen.isChecked(),
            use_osc=self.check_osc.isChecked(),
            use_sa=self.check_sa.isChecked(),
            use_pm=self.check_pm.isChecked(),
            use_lsa=self.check_lsa.isChecked()
        )

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setMaximum(len(vals))
        self.progress_bar.setValue(0)
        self.start_requested.emit(task)

    def reset_controls(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)


# ==============================================================================
# ПОТОК ВЫПОЛНЕНИЯ ЭКСПЕРИМЕНТА
# ==============================================================================
class ExperimentThread(QThread):
    log_msg = pyqtSignal(str)
    progress = pyqtSignal(int)
    finished_experiment = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, drivers: Dict[str, Any], task: ExperimentTask):
        super().__init__()
        self.drivers = drivers
        self.task = task
        self._is_running = True

    def stop(self):
        self._is_running = False

    def _setup_folders(self):
        date_str = datetime.now().strftime("%Y%m%d")
        self.exp_folder = os.path.join(self.task.save_dir, date_str)
        os.makedirs(self.exp_folder, exist_ok=True)
        self.folders = {}

        if self.task.use_gen:
            self.folders['gen'] = os.path.join(self.exp_folder, "Generator")
            os.makedirs(self.folders['gen'], exist_ok=True)

        if self.task.use_osc:
            self.folders['osc_wave'] = os.path.join(self.exp_folder, "Oscilloscope", "Waveforms")
            self.folders['osc_meas'] = os.path.join(self.exp_folder, "Oscilloscope", "Measurements")
            self.folders['osc_screen'] = os.path.join(self.exp_folder, "Oscilloscope", "Screenshots")
            os.makedirs(self.folders['osc_wave'], exist_ok=True)
            os.makedirs(self.folders['osc_meas'], exist_ok=True)
            os.makedirs(self.folders['osc_screen'], exist_ok=True)

        if self.task.use_sa:
            self.folders['sa_traces'] = os.path.join(self.exp_folder, "SpectrumAnalyzer_DSA", "Traces")
            os.makedirs(self.folders['sa_traces'], exist_ok=True)

        if self.task.use_pm:
            self.folders['pm'] = os.path.join(self.exp_folder, "PowerMeter")
            os.makedirs(self.folders['pm'], exist_ok=True)

        if self.task.use_lsa:
            self.folders['lsa_smx'] = os.path.join(self.exp_folder, "OpticalAnalyzer_LSA", "SMX_Raw")
            self.folders['lsa_csv'] = os.path.join(self.exp_folder, "OpticalAnalyzer_LSA", "CSV_Export")
            os.makedirs(self.folders['lsa_smx'], exist_ok=True)
            os.makedirs(self.folders['lsa_csv'], exist_ok=True)

    def _init_log_files(self):
        self.pm_log_file = ""
        self.lsa_log_file = ""
        if self.task.use_pm and 'pm' in self.folders:
            n = 1
            while True:
                path = os.path.join(self.folders['pm'], f"power_log_exp{n:03d}.csv")
                if not os.path.exists(path):
                    self.pm_log_file = path
                    break
                n += 1

        if self.task.use_lsa and 'lsa_csv' in self.folders:
            n = 1
            while True:
                path = os.path.join(os.path.dirname(self.folders['lsa_csv']), f"lsa_log_exp{n:03d}.csv")
                if not os.path.exists(path):
                    self.lsa_log_file = path
                    break
                n += 1

    def run(self):
        try:
            self.log_msg.emit("Подготовка окружения эксперимента...")
            self._setup_folders()
            self._init_log_files()

            gen: Optional[RigolGeneratorDriver] = self.drivers.get('gen')
            osc: Optional[RigolOscilloscopeDriver] = self.drivers.get('osc')
            sa: Optional[SpectrumAnalyzerDriver] = self.drivers.get('sa')
            pm: Optional[PowerMeterDriver] = self.drivers.get('pm')
            lsa: Optional[LSADriver] = self.drivers.get('lsa')

            # Сохранение начального профиля при ручном режиме (1 раз)
            if self.task.use_osc and osc and not self.task.osc_auto:
                osc.save_state_json(self.exp_folder, -1)
            if self.task.use_sa and sa and not self.task.sa_auto:
                sa.save_state_json(self.exp_folder, -1, mode_name="manual")

            if self.task.use_gen and gen:
                if self.task.target_ch in ["Канал 1", "Оба канала"]:
                    gen.set_output(1, True)
                if self.task.target_ch in ["Канал 2", "Оба канала"]:
                    gen.set_output(2, True)

            num_steps = len(self.task.values)

            for i in range(num_steps):
                if not self._is_running:
                    self.log_msg.emit("Эксперимент остановлен пользователем.")
                    break

                step_num = i + 1
                cur_val = self.task.values[i]

                if self.task.mode == "DC":
                    cur_volt = float(cur_val)
                    cur_freq = self.task.sa_gen_freq
                else:
                    cur_freq = float(cur_val[0])
                    cur_volt = float(cur_val[1])

                self.log_msg.emit(f"Шаг {step_num}/{num_steps}: Напряжение={cur_volt:.3f} В")

                # 1. Генератор
                if self.task.use_gen and gen:
                    try:
                        if self.task.mode == "DC":
                            if self.task.target_ch in ["Канал 1", "Оба канала"]:
                                gen.safe_write(':SOUR1:FUNC DC')
                                gen.safe_write(f':SOUR1:VOLT:OFFS {cur_volt}')
                                gen.safe_write(':SOUR1:VOLT 0')
                            if self.task.target_ch in ["Канал 2", "Оба канала"]:
                                gen.safe_write(':SOUR2:FUNC DC')
                                gen.safe_write(f':SOUR2:VOLT:OFFS {cur_volt}')
                                gen.safe_write(':SOUR2:VOLT 0')
                        elif self.task.mode == "SIN параметры":
                            f, h, l = cur_val
                            if self.task.target_ch in ["Канал 1", "Оба канала"]:
                                gen.safe_write(':SOUR1:FUNC SIN')
                                gen.safe_write(f':SOUR1:FREQ {f}')
                                gen.safe_write(f':SOUR1:VOLT:HIGH {h}')
                                gen.safe_write(f':SOUR1:VOLT:LOW {l}')
                            if self.task.target_ch in ["Канал 2", "Оба канала"]:
                                gen.safe_write(':SOUR2:FUNC SIN')
                                gen.safe_write(f':SOUR2:FREQ {f}')
                                gen.safe_write(f':SOUR2:VOLT:HIGH {h}')
                                gen.safe_write(f':SOUR2:VOLT:LOW {l}')
                        else:
                            f, h, l, w = cur_val
                            if self.task.target_ch in ["Канал 1", "Оба канала"]:
                                gen.safe_write(':SOUR1:FUNC PULS')
                                gen.safe_write(f':SOUR1:FREQ {f}')
                                gen.safe_write(f':SOUR1:VOLT:HIGH {h}')
                                gen.safe_write(f':SOUR1:VOLT:LOW {l}')
                                gen.safe_write(f':SOUR1:PULS:WIDT {w}')
                            if self.task.target_ch in ["Канал 2", "Оба канала"]:
                                gen.safe_write(':SOUR2:FUNC PULS')
                                gen.safe_write(f':SOUR2:FREQ {f}')
                                gen.safe_write(f':SOUR2:VOLT:HIGH {h}')
                                gen.safe_write(f':SOUR2:VOLT:LOW {l}')
                                gen.safe_write(f':SOUR2:PULS:WIDT {w}')

                        gen.save_state_json(self.exp_folder, step_num)
                    except Exception as e:
                        self.log_msg.emit(f"Сбой генератора: {e}")

                # 2. Автонастройка спектроанализатора под сигнал
                if self.task.use_sa and sa and self.task.sa_auto:
                    try:
                        span = max(cur_freq * 0.1, 100000.0)
                        rbw = max(span * 0.01, 1000.0)
                        sa.apply_settings(center_freq=cur_freq, span=span, rbw=rbw, vbw=rbw)
                        sa.save_state_json(self.exp_folder, step_num, mode_name="auto")
                    except Exception as e:
                        self.log_msg.emit(f"Сбой автонастройки DSA: {e}")

                # 3. Автонастройка осциллографа под сигнал
                if self.task.use_osc and osc and self.task.osc_auto:
                    try:
                        osc.autoscale()
                        osc.save_state_json(self.exp_folder, step_num)
                    except Exception as e:
                        self.log_msg.emit(f"Сбой автонастройки осциллографа: {e}")

                time.sleep(self.task.pause_s)

                # 4. Измеритель мощности
                if self.task.use_pm and pm and self.pm_log_file:
                    try:
                        pm_data = pm.measure_power()
                        current_ma = cur_volt * 150.0
                        df_pm = pd.DataFrame([{
                            'Step': step_num,
                            'Voltage_V': cur_volt,
                            'Current_mA': current_ma,
                            'Power_W': pm_data['power_w'],
                            'Power_dBm': pm_data['power_dbm'],
                            'PM_Raw_Unit': pm_data['raw_unit'],
                            'Time': datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        }])
                        m = 'a' if os.path.exists(self.pm_log_file) else 'w'
                        df_pm.to_csv(self.pm_log_file, mode=m, header=not os.path.exists(self.pm_log_file),
                                     index=False, sep=';', decimal=',')
                        self.log_msg.emit(f"  PM: {pm_data['power_w'] * 1e3:.2f} мВт ({pm_data['power_dbm']:.2f} дБм)")
                    except Exception as e:
                        self.log_msg.emit(f"Сбой PM: {e}")

                # 5. LSA
                if self.task.use_lsa and lsa and self.lsa_log_file:
                    try:
                        v_str = f"{cur_volt:.2f}".replace('.', '_')
                        smx_path = os.path.join(self.folders['lsa_smx'], f"LSA_step{step_num:03d}_{v_str}V.smx")
                        csv_path = os.path.join(self.folders['lsa_csv'], f"LSA_step{step_num:03d}_{v_str}V.csv")
                        peak_nm = lsa.save_smx_and_get_peak(smx_path)
                        lsa.convert_smx_to_csv(smx_path, csv_path)

                        df_lsa = pd.DataFrame([{
                            'Step': step_num,
                            'Voltage_V': cur_volt,
                            'LSA_Peak_nm': peak_nm,
                            'Time': datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        }])
                        m = 'a' if os.path.exists(self.lsa_log_file) else 'w'
                        df_lsa.to_csv(self.lsa_log_file, mode=mode, header=not os.path.exists(self.lsa_log_file),
                                      index=False, sep=';', decimal=',')
                        self.log_msg.emit(f"  LSA: Пик={peak_nm:.4f} нм сохранен")
                    except Exception as e:
                        self.log_msg.emit(f"Сбой LSA: {e}")

                # 6. Осциллограф (сбор форм волн)
                if self.task.use_osc and osc:
                    try:
                        if self.task.osc_ch1_enabled:
                            m1 = osc.save_channel_data(1, step_num, cur_volt,
                                                       self.folders['osc_wave'], self.folders['osc_meas'],
                                                       self.folders['osc_screen'])
                            self.log_msg.emit(f"  OSC CH1: VRMS={m1['vrms']:.3f} В")
                        if self.task.osc_ch2_enabled:
                            m2 = osc.save_channel_data(2, step_num, cur_volt,
                                                       self.folders['osc_wave'], self.folders['osc_meas'],
                                                       self.folders['osc_screen'])
                            self.log_msg.emit(f"  OSC CH2: VRMS={m2['vrms']:.3f} В")
                        osc.run()
                    except Exception as e:
                        self.log_msg.emit(f"Сбой OSC: {e}")

                # 7. Спектроанализатор (сбор трасс)
                if self.task.use_sa and sa:
                    try:
                        sa.save_trace_csv(self.folders['sa_traces'], step_num, cur_volt, cur_freq)
                        self.log_msg.emit("  DSA: Трасса сохранена")
                    except Exception as e:
                        self.log_msg.emit(f"Сбой DSA: {e}")

                self.progress.emit(step_num)

            self.log_msg.emit("Эксперимент полностью завершен.")
        except Exception as e:
            self.error_occurred.emit(str(e))
            traceback.print_exc()
        finally:
            if self.task.use_gen:
                gen_drv = self.drivers.get('gen')
                if gen_drv:
                    try:
                        gen_drv.set_output(1, False)
                        gen_drv.set_output(2, False)
                        self.log_msg.emit("Безопасность: выходы генератора переведены в состояние ВЫКЛ.")
                    except Exception as e_gen:
                        self.log_msg.emit(f"Критический сбой отключения генератора: {e_gen}")

            self.finished_experiment.emit()


# ==============================================================================
# БЕЗОПАСНЫЙ ПОТОК ОБНАРУЖЕНИЯ ПРИБОРОВ (БЕЗ СБРОСА)
# ==============================================================================
class DeviceScanThread(QThread):
    device_found = pyqtSignal(str, str, object)
    error_occurred = pyqtSignal(str)
    finished_scan = pyqtSignal()

    def __init__(self, rm):
        super().__init__()
        self.rm = rm

    def run(self):
        try:
            resources = self.rm.list_resources()
            if not resources:
                self.error_occurred.emit("Устройства VISA не обнаружены.")
                return

            for res in resources:
                if 'ASRL10' in res or 'ASRL11' in res:
                    continue

                res_u = res.upper()
                name = "Неизвестное устройство"
                drv = None

                try:
                    raw_dev = self.rm.open_resource(res)
                    raw_dev.timeout = 3000
                    
                    # Безопасная проверка модели без отправки сбросов или команд развертки
                    idn = ""
                    try:
                        idn = raw_dev.query('*IDN?').strip()
                    except Exception:
                        pass

                    if 'DG' in res_u or 'DG' in idn:
                        name = f"Генератор ({idn.split(',')[1] if ',' in idn else 'Rigol DG'})"
                        drv = RigolGeneratorDriver(raw_dev)
                    elif 'DS' in res_u or 'MSO' in res_u or 'DS' in idn:
                        name = f"Осциллограф ({idn.split(',')[1] if ',' in idn else 'Rigol DS'})"
                        drv = RigolOscilloscopeDriver(raw_dev)
                    elif 'DSA' in res_u or 'DSA' in idn:
                        name = f"Спектроанализатор ({idn.split(',')[1] if ',' in idn else 'Rigol DSA'})"
                        drv = SpectrumAnalyzerDriver(raw_dev)
                    elif 'ASRL' in res_u or 'PM' in res_u:
                        name = "Измеритель мощности (PM100USB)"
                        drv = PowerMeterDriver(raw_dev)
                    else:
                        raw_dev.close()
                except Exception as e:
                    name += f" (Сбой: {e})"

                self.device_found.emit(name, res, drv)
        except Exception as e:
            self.error_occurred.emit(f"Ошибка сканирования: {e}")
        finally:
            self.finished_scan.emit()


# ==============================================================================
# ГЛАВНОЕ ОКНО ПРИЛОЖЕНИЯ
# ==============================================================================
class RigolControlApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rigol Automated Measurement Suite")
        self.setMinimumSize(950, 750)

        self.rm = pyvisa.ResourceManager()
        self.gen_driver: Optional[RigolGeneratorDriver] = None
        self.osc_driver: Optional[RigolOscilloscopeDriver] = None
        self.sa_driver: Optional[SpectrumAnalyzerDriver] = None
        self.pm_driver: Optional[PowerMeterDriver] = None
        self.lsa_driver: LSADriver = LSADriver()

        self.scan_thread: Optional[DeviceScanThread] = None
        self.exp_thread: Optional[ExperimentThread] = None

        self.init_ui()
        QTimer.singleShot(600, self.start_scan)

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        conn_box = QGroupBox("Обнаружение приборов (безопасное обновление без сброса)")
        conn_l = QVBoxLayout(conn_box)
        top_c = QHBoxLayout()
        self.btn_scan = QPushButton("Обновить устройства")
        self.btn_scan.clicked.connect(self.start_scan)
        top_c.addWidget(self.btn_scan)
        top_c.addStretch()
        conn_l.addLayout(top_c)

        self.list_devices = QListWidget()
        self.list_devices.setMaximumHeight(90)
        conn_l.addWidget(self.list_devices)
        layout.addWidget(conn_box)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # 1. Генератор
        tab_gen = QWidget()
        gen_l = QVBoxLayout(tab_gen)
        ch_l = QHBoxLayout()
        self.panel_ch1 = ChannelPanel(1)
        self.panel_ch2 = ChannelPanel(2)
        self.panel_ch1.output_toggled_signal.connect(self.toggle_channel_output)
        self.panel_ch2.output_toggled_signal.connect(self.toggle_channel_output)
        ch_l.addWidget(self.panel_ch1)
        ch_l.addWidget(self.panel_ch2)
        gen_l.addLayout(ch_l)

        btn_bar = QHBoxLayout()
        b_ch1 = QPushButton("Применить Канал 1")
        b_ch2 = QPushButton("Применить Канал 2")
        b_both = QPushButton("Применить оба")
        b_ch1.clicked.connect(lambda: self.apply_channel(1))
        b_ch2.clicked.connect(lambda: self.apply_channel(2))
        b_both.clicked.connect(self.apply_both_channels)

        btn_bar.addWidget(b_ch1)
        btn_bar.addWidget(b_ch2)
        btn_bar.addWidget(b_both)
        gen_l.addLayout(btn_bar)
        gen_l.addStretch()
        self.tabs.addTab(tab_gen, "Генератор")

        # 2. Осциллограф
        tab_osc = QWidget()
        osc_l = QVBoxLayout(tab_osc)
        self.panel_osc = OscilloscopePanel()
        osc_l.addWidget(self.panel_osc)

        btn_osc_apply = QPushButton("Применить параметры развертки")
        btn_osc_apply.clicked.connect(self.apply_oscilloscope)
        osc_l.addWidget(btn_osc_apply)

        osc_ctrl = QHBoxLayout()
        b_auto = QPushButton("Автомасштаб (:AUT)")
        b_run = QPushButton("Пуск")
        b_stop = QPushButton("Стоп")
        b_auto.clicked.connect(lambda: self.osc_driver.autoscale() if self.osc_driver else None)
        b_run.clicked.connect(lambda: self.osc_driver.run() if self.osc_driver else None)
        b_stop.clicked.connect(lambda: self.osc_driver.stop() if self.osc_driver else None)
        osc_ctrl.addWidget(b_auto)
        osc_ctrl.addWidget(b_run)
        osc_ctrl.addWidget(b_stop)
        osc_l.addLayout(osc_ctrl)
        osc_l.addStretch()
        self.tabs.addTab(tab_osc, "Осциллограф")

        # 3. Спектроанализатор
        tab_sa = QWidget()
        sa_l = QVBoxLayout(tab_sa)
        self.panel_sa = SpectrumAnalyzerPanel()
        sa_l.addWidget(self.panel_sa)

        b_sa_apply = QPushButton("Применить параметры SPAN/RBW")
        b_sa_apply.clicked.connect(self.apply_sa)
        sa_l.addWidget(b_sa_apply)

        sa_ctrl = QHBoxLayout()
        b_peak = QPushButton("Найти пик")
        b_peak.clicked.connect(self.sa_find_peak)
        sa_ctrl.addWidget(b_peak)
        sa_l.addLayout(sa_ctrl)
        sa_l.addStretch()
        self.tabs.addTab(tab_sa, "Спектроанализатор")

        # 4. Мощность и Оптика
        self.panel_optics = OpticsPowerPanel()
        self.panel_optics.manual_power_requested.connect(self.manual_power_measure)
        self.panel_optics.convert_smx_requested.connect(self.manual_smx_convert)
        self.tabs.addTab(self.panel_optics, "Мощность и Оптика")

        # 5. Эксперимент
        self.panel_exp = ExperimentPanel()
        self.panel_exp.start_requested.connect(self.start_experiment)
        self.panel_exp.stop_requested.connect(self.stop_experiment)
        self.tabs.addTab(self.panel_exp, "Автоматизация (Эксперимент)")

    def toggle_channel_output(self, ch: int, state: bool):
        if self.gen_driver:
            self.gen_driver.set_output(ch, state)

    def apply_channel(self, ch: int):
        if not self.gen_driver:
            QMessageBox.warning(self, "Ошибка", "Генератор не подключен.")
            return
        panel = self.panel_ch1 if ch == 1 else self.panel_ch2
        self.gen_driver.apply_channel_state(ch, panel.get_state())

    def apply_both_channels(self):
        self.apply_channel(1)
        self.apply_channel(2)

    def apply_oscilloscope(self):
        if not self.osc_driver:
            QMessageBox.warning(self, "Ошибка", "Осциллограф не подключен.")
            return
        self.osc_driver.apply_state(self.panel_osc.get_state())
        QMessageBox.information(self, "Успех", "Параметры осциллографа переданы.")

    def apply_sa(self):
        if not self.sa_driver:
            QMessageBox.warning(self, "Ошибка", "Спектроанализатор не подключен.")
            return
        st = self.panel_sa.get_state()
        self.sa_driver.apply_settings(st.center_freq, st.span, st.rbw, st.vbw)
        QMessageBox.information(self, "Успех", "Параметры спектроанализатора переданы.")

    def sa_find_peak(self):
        if not self.sa_driver:
            return
        f, a = self.sa_driver.find_peak()
        QMessageBox.information(self, "Пик DSA", f"Частота: {f:.2f} Гц\nАмплитуда: {a:.2f} дБм")

    def manual_power_measure(self):
        if not self.pm_driver:
            QMessageBox.warning(self, "Ошибка", "Измеритель мощности не подключен.")
            return
        try:
            res = self.pm_driver.measure_power()
            self.panel_optics.update_power_result(res)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))
            self.panel_optics.update_power_result(None)

    def manual_smx_convert(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Выберите файлы SMX", "", "SMX Files (*.smx)")
        if not files:
            return
        cnt = 0
        for f in files:
            csv_path = os.path.splitext(f)[0] + ".csv"
            try:
                self.lsa_driver.convert_smx_to_csv(f, csv_path)
                cnt += 1
            except Exception as e:
                print(f"Ошибка конвертации {f}: {e}")
        QMessageBox.information(self, "Конвертация", f"Успешно обработано: {cnt} из {len(files)}")

    def close_all_drivers(self):
        if self.gen_driver:
            self.gen_driver.close()
            self.gen_driver = None
        if self.osc_driver:
            self.osc_driver.close()
            self.osc_driver = None
        if self.sa_driver:
            self.sa_driver.close()
            self.sa_driver = None
        if self.pm_driver:
            self.pm_driver.close()
            self.pm_driver = None

        try:
            self.rm.close()
        except Exception:
            pass
        self.rm = pyvisa.ResourceManager()

    def start_scan(self):
        if self.scan_thread and self.scan_thread.isRunning():
            return

        self.list_devices.clear()
        self.btn_scan.setEnabled(False)
        self.btn_scan.setText("Опрос шины...")
        QApplication.processEvents()

        self.close_all_drivers()
        time.sleep(0.2)

        self.scan_thread = DeviceScanThread(self.rm)
        self.scan_thread.device_found.connect(self.on_device_found)
        self.scan_thread.error_occurred.connect(lambda err: self.list_devices.addItem(f"Ошибка: {err}"))
        self.scan_thread.finished_scan.connect(self.on_scan_finished)
        self.scan_thread.start()

    def on_device_found(self, name: str, port: str, driver: Any):
        item = QListWidgetItem(f"{name} [{port}]")
        if "Неизвестное" not in name and "Сбой" not in name:
            item.setForeground(QColor("green"))
        self.list_devices.addItem(item)

        if isinstance(driver, RigolGeneratorDriver): self.gen_driver = driver
        elif isinstance(driver, RigolOscilloscopeDriver): self.osc_driver = driver
        elif isinstance(driver, SpectrumAnalyzerDriver): self.sa_driver = driver
        elif isinstance(driver, PowerMeterDriver): self.pm_driver = driver

    def on_scan_finished(self):
        self.btn_scan.setEnabled(True)
        self.btn_scan.setText("Обновить устройства")

    def start_experiment(self, task: ExperimentTask):
        osc_state = self.panel_osc.get_state()
        task.osc_ch1_enabled = osc_state.ch1_enabled
        task.osc_ch2_enabled = osc_state.ch2_enabled
        
        # Получаем флаги автонастройки из выпадающих списков
        task.osc_auto = self.panel_osc.is_auto_scale_enabled()
        task.sa_auto = self.panel_sa.is_auto_tune_enabled()

        if task.use_osc and not task.osc_ch1_enabled and not task.osc_ch2_enabled:
            QMessageBox.warning(self, "Внимание", "На панели осциллографа выключены оба канала! Запись форм не выполнится.")

        drivers = {
            'gen': self.gen_driver,
            'osc': self.osc_driver,
            'sa': self.sa_driver,
            'pm': self.pm_driver,
            'lsa': self.lsa_driver
        }
        self.exp_thread = ExperimentThread(drivers, task)
        self.exp_thread.log_msg.connect(self.panel_exp.log)
        self.exp_thread.progress.connect(self.panel_exp.progress_bar.setValue)
        self.exp_thread.finished_experiment.connect(self.panel_exp.reset_controls)
        self.exp_thread.error_occurred.connect(lambda err: QMessageBox.critical(self, "Ошибка эксперимента", err))
        self.exp_thread.start()

    def stop_experiment(self):
        if self.exp_thread and self.exp_thread.isRunning():
            self.exp_thread.stop()
            self.panel_exp.log("Прерывание: остановка после завершения текущей точки...")

    def closeEvent(self, event):
        if self.exp_thread and self.exp_thread.isRunning():
            self.exp_thread.stop()
            self.exp_thread.wait(2000)

        self.close_all_drivers()
        event.accept()


# ==============================================================================
# ТОЧКА ВХОДА И ЗАПУСК UDP-МОСТА
# ==============================================================================
if __name__ == '__main__':
    if getattr(sys, 'frozen', False):
        bundle_dir = sys._MEIPASS
        exe_dir = os.path.dirname(sys.executable)
    else:
        bundle_dir = os.path.dirname(os.path.abspath(__file__))
        exe_dir = bundle_dir

    bridge_script = os.path.join(bundle_dir, "lsa_bridge.py")
    python32_exe = os.path.join(exe_dir, "python_x86", "python.exe")

    if not os.path.exists(python32_exe):
        python32_exe = os.path.join(bundle_dir, "python_x86", "python.exe")
        if not os.path.exists(python32_exe):
            python32_exe = "python"

    if os.path.exists(bridge_script):
        try:
            creation_flags = 0x08000000 if sys.platform == "win32" else 0
            bridge_process = subprocess.Popen(
                [python32_exe, bridge_script],
                cwd=bundle_dir,
                creationflags=creation_flags
            )
            atexit.register(bridge_process.terminate)
            print("32-битный UDP мост LSA запущен.")
        except Exception as e:
            print(f"Ошибка запуска моста LSA: {e}")

    app = QApplication(sys.argv)
    window = RigolControlApp()
    window.show()
    sys.exit(app.exec_())