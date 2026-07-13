# -*- coding: utf-8 -*-
"""Generate demo wav files into ./demo : sweep, recorded sweep, audition."""
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

import ir_core

OUT = Path(__file__).parent / "demo"
rng = np.random.default_rng(7)


def demo_room_ir(sr, seconds=1.2):
    """Synthetic stereo room: direct + early reflections + diffuse tail."""
    n = int(sr * seconds)
    ir = np.zeros((n, 2))
    early = [(0.0, 1.0), (11.3, 0.42), (17.9, -0.31), (29.5, 0.26),
             (43.1, -0.18), (61.7, 0.12)]
    for c in range(2):
        for ms, g in early:
            i = int(sr * ms / 1000 * (1 + 0.06 * c))
            ir[i, c] += g
        t = np.arange(n) / sr
        tail = rng.standard_normal(n) * np.exp(-t / 0.35) * 0.12
        sos = signal.butter(2, [120 / (sr / 2), 7500 / (sr / 2)],
                            "bandpass", output="sos")
        ir[:, c] += signal.sosfilt(sos, tail)
    sos = signal.butter(6, 15000 / (sr / 2), "lowpass", output="sos")
    ir = signal.sosfiltfilt(sos, ir, axis=0)
    return ir / np.abs(ir).max()


def demo_audition(sr, seconds=6.0):
    """A dry pluck/drum pattern to audition the IR with."""
    n = int(sr * seconds)
    x = np.zeros(n)
    t_note = np.arange(int(sr * 0.5)) / sr
    for beat, f0 in enumerate([220, 220, 330, 220, 262, 220, 196, 165]):
        start = int(beat * 0.7 * sr)
        pluck = np.sin(2 * np.pi * f0 * t_note) * 0.5
        pluck += np.sin(2 * np.pi * 2 * f0 * t_note) * 0.25
        pluck += np.sin(2 * np.pi * 3 * f0 * t_note) * 0.12
        pluck *= np.exp(-t_note / 0.15)
        click = rng.standard_normal(len(t_note)) * np.exp(-t_note / 0.005) * 0.4
        seg = pluck + click
        end = min(start + len(seg), n)
        x[start:end] += seg[:end - start]
    return (0.7 * x / np.abs(x).max())[:, None]


def main():
    OUT.mkdir(exist_ok=True)
    sr = 48000
    sweep = ir_core.make_ess(20, 20000, 6.0, sr)[:, None] * 0.7
    sf.write(OUT / "sweep_48k.wav", sweep, sr, subtype="PCM_24")

    room = demo_room_ir(sr)
    rec = np.column_stack([
        signal.fftconvolve(sweep[:, 0], room[:, c]) for c in range(2)])
    rec = np.pad(rec, ((int(0.5 * sr), int(0.5 * sr)), (0, 0)))
    rec += rng.standard_normal(rec.shape) * 1e-4      # mic noise floor
    rec *= 0.6 / np.abs(rec).max()
    sf.write(OUT / "recorded_48k.wav", rec, sr, subtype="PCM_24")

    aud = demo_audition(sr)
    sf.write(OUT / "audition_48k.wav", aud, sr, subtype="PCM_16")

    sf.write(OUT / "true_room_ir.wav", room, sr, subtype="FLOAT")
    print("demo files written to", OUT)


if __name__ == "__main__":
    main()
