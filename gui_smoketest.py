# -*- coding: utf-8 -*-
"""Drive the GUI end-to-end without user interaction (needs a display)."""
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import tempfile
from pathlib import Path

import tkinter as tk
from tkinter import ttk

import soundfile as sf

import irpond

DEMO = Path(__file__).parent / "demo"


def pump(root, seconds):
    end = time.time() + seconds
    while time.time() < end:
        root.update()
        time.sleep(0.02)


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    app = irpond.App(root)
    root.update()

    # --- create IR (measured mode) ---
    app.var_src.set(str(DEMO / "sweep_48k.wav"))
    app.var_rec.set(str(DEMO / "recorded_48k.wav"))
    app.var_len.set("1.0")
    app.on_make()
    for _ in range(600):                     # wait up to 60 s
        pump(root, 0.1)
        if not app.busy:
            break
    assert app.ir is not None, "IR not created (measured mode)"
    assert app.ir.shape[1] == 2, f"expected stereo IR, got {app.ir.shape}"
    print("measured-mode IR ok:", app.ir.shape, app.ir_sr, "Hz")

    # --- create IR (blind mode) ---
    app.var_src.set("")
    app.on_make()
    for _ in range(3000):
        pump(root, 0.1)
        if not app.busy:
            break
    assert app.ir is not None, "IR not created (blind mode)"
    print("blind-mode IR ok:", app.var_status.get().replace("\n", " | "))

    # --- audition: render + play a moment, adjust wet, stop ---
    app.var_aud.set(str(DEMO / "audition_48k.wav"))
    app.on_play()
    for _ in range(300):
        pump(root, 0.1)
        if not app.busy:
            break
    played = app.player.playing
    app.var_wet.set(80.0)
    app._on_wet()
    pump(root, 1.0)
    app.on_stop()
    print("playback started:", played, "| wet now", app.player.wet)

    # --- save ---
    out = Path(tempfile.mkdtemp(prefix="irpond_gui_")) / "ir.wav"
    app.var_sr.set("96000")
    app.var_bit.set("32-bit float")
    app.var_ch.set("モノラル")
    import tkinter.filedialog as fd
    fd.asksaveasfilename = lambda **kw: str(out)
    app.on_save()
    info = sf.info(str(out))
    assert info.samplerate == 96000 and info.channels == 1 \
        and info.subtype == "FLOAT", info
    print("save ok:", info.samplerate, "Hz", info.channels, "ch", info.subtype)

    root.destroy()
    print("GUI smoke test passed")


if __name__ == "__main__":
    main()
