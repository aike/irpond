# -*- coding: utf-8 -*-
"""irpond — Impulse Response maker GUI.

Sweep measurement -> IR estimation -> audition with wet control -> save.
"""
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import sounddevice as sd

import ir_core

SR_CHOICES = ["元のまま", "44100", "48000", "88200", "96000", "192000"]
BIT_CHOICES = list(ir_core.SUBTYPES.keys())          # 16/24-bit PCM, 32-bit float
CH_CHOICES = ["元のまま", "モノラル", "ステレオ"]
LEN_CHOICES = ["自動", "0.1", "0.2", "0.5", "1.0", "2.0", "5.0", "10.0"]


class Player:
    """Streaming player mixing dry/wet buffers with a live wet ratio."""

    def __init__(self):
        self.stream = None
        self.dry = None
        self.wet_sig = None
        self.pos = 0
        self.wet = 0.5

    def play(self, dry, wet_sig, sr):
        self.stop()
        self.dry, self.wet_sig, self.pos = dry, wet_sig, 0
        n, ch = dry.shape

        def callback(out, frames, time_info, status):
            i = self.pos
            j = min(i + frames, n)
            k = j - i
            w = min(max(self.wet, 0.0), 1.0)
            out[:k] = (1.0 - w) * self.dry[i:j] + w * self.wet_sig[i:j]
            if k < frames:
                out[k:] = 0.0
                raise sd.CallbackStop
            self.pos = j

        self.stream = sd.OutputStream(samplerate=sr, channels=ch,
                                      dtype="float32", callback=callback)
        self.stream.start()

    def stop(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    @property
    def playing(self):
        return self.stream is not None and self.stream.active


class App:
    def __init__(self, root):
        self.root = root
        root.title("irpond — Impulse Response メーカー")
        root.minsize(660, 560)

        self.ir = None            # (n, ch) float64
        self.ir_sr = None
        self.player = Player()
        self.jobs = queue.Queue()
        self.busy = False

        pad = {"padx": 8, "pady": 4}
        body = ttk.Frame(root, padding=8)
        body.pack(fill="both", expand=True)

        # ---------------- input files ----------------
        f_in = ttk.LabelFrame(body, text=" 入力ファイル ", padding=6)
        f_in.pack(fill="x", **pad)
        f_in.columnconfigure(1, weight=1)

        ttk.Label(f_in, text="スイープ原音 wav:").grid(row=0, column=0, sticky="w")
        self.var_src = tk.StringVar()
        ttk.Entry(f_in, textvariable=self.var_src).grid(
            row=0, column=1, sticky="ew", padx=4)
        ttk.Button(f_in, text="参照...",
                   command=lambda: self._browse(self.var_src)).grid(row=0, column=2)
        ttk.Button(f_in, text="クリア",
                   command=lambda: self.var_src.set("")).grid(
            row=0, column=3, padx=(4, 0))
        ttk.Label(f_in, foreground="#666",
                  text="※ 空欄の場合はターゲット録音からスイープを自動推定します"
                  ).grid(row=1, column=1, columnspan=3, sticky="w", padx=4)

        ttk.Label(f_in, text="ターゲット録音 wav:").grid(row=2, column=0, sticky="w")
        self.var_rec = tk.StringVar()
        ttk.Entry(f_in, textvariable=self.var_rec).grid(
            row=2, column=1, sticky="ew", padx=4)
        ttk.Button(f_in, text="参照...",
                   command=lambda: self._browse(self.var_rec)).grid(row=2, column=2)

        # ---------------- IR creation ----------------
        f_ir = ttk.LabelFrame(body, text=" IR 作成 ", padding=6)
        f_ir.pack(fill="x", **pad)

        row = ttk.Frame(f_ir)
        row.pack(fill="x")
        ttk.Label(row, text="IR 長さ(秒):").pack(side="left")
        self.var_len = tk.StringVar(value="自動")
        ttk.Combobox(row, textvariable=self.var_len, values=LEN_CHOICES,
                     width=7, state="readonly").pack(side="left", padx=4)
        self.var_norm = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text="正規化 (-1 dBFS)",
                        variable=self.var_norm).pack(side="left", padx=12)
        self.btn_make = ttk.Button(row, text="IR を作成", command=self.on_make)
        self.btn_make.pack(side="left", padx=12)

        self.var_status = tk.StringVar(value="ファイルを選択して「IR を作成」を押してください")
        ttk.Label(f_ir, textvariable=self.var_status, foreground="#06c",
                  wraplength=600, justify="left").pack(fill="x", pady=(6, 0))

        self.canvas = tk.Canvas(f_ir, height=130, bg="#101418",
                                highlightthickness=1,
                                highlightbackground="#888")
        self.canvas.pack(fill="x", pady=(6, 0))
        self.canvas.bind("<Configure>", lambda e: self._draw_ir())

        # ---------------- audition ----------------
        f_pv = ttk.LabelFrame(body, text=" 試聴 ", padding=6)
        f_pv.pack(fill="x", **pad)
        f_pv.columnconfigure(1, weight=1)

        ttk.Label(f_pv, text="試聴用音源 wav:").grid(row=0, column=0, sticky="w")
        self.var_aud = tk.StringVar()
        ttk.Entry(f_pv, textvariable=self.var_aud).grid(
            row=0, column=1, sticky="ew", padx=4)
        ttk.Button(f_pv, text="参照...",
                   command=lambda: self._browse(self.var_aud)).grid(row=0, column=2)

        row2 = ttk.Frame(f_pv)
        row2.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        row2.columnconfigure(1, weight=1)
        ttk.Label(row2, text="Wet:").grid(row=0, column=0)
        self.var_wet = tk.DoubleVar(value=50.0)
        self.scale_wet = ttk.Scale(row2, from_=0, to=100, variable=self.var_wet,
                                   command=self._on_wet)
        self.scale_wet.grid(row=0, column=1, sticky="ew", padx=6)
        self.var_wet_lbl = tk.StringVar(value="50 %")
        ttk.Label(row2, textvariable=self.var_wet_lbl, width=6).grid(
            row=0, column=2)
        self.btn_play = ttk.Button(row2, text="▶ 再生", command=self.on_play)
        self.btn_play.grid(row=0, column=3, padx=(10, 2))
        self.btn_stop = ttk.Button(row2, text="■ 停止", command=self.on_stop,
                                   state="disabled")
        self.btn_stop.grid(row=0, column=4)

        # ---------------- save ----------------
        f_sv = ttk.LabelFrame(body, text=" IR 保存 ", padding=6)
        f_sv.pack(fill="x", **pad)
        row3 = ttk.Frame(f_sv)
        row3.pack(fill="x")
        ttk.Label(row3, text="サンプルレート:").pack(side="left")
        self.var_sr = tk.StringVar(value="元のまま")
        ttk.Combobox(row3, textvariable=self.var_sr, values=SR_CHOICES,
                     width=9, state="readonly").pack(side="left", padx=(2, 10))
        ttk.Label(row3, text="量子化:").pack(side="left")
        self.var_bit = tk.StringVar(value="24-bit PCM")
        ttk.Combobox(row3, textvariable=self.var_bit, values=BIT_CHOICES,
                     width=12, state="readonly").pack(side="left", padx=(2, 10))
        ttk.Label(row3, text="チャンネル:").pack(side="left")
        self.var_ch = tk.StringVar(value="元のまま")
        ttk.Combobox(row3, textvariable=self.var_ch, values=CH_CHOICES,
                     width=9, state="readonly").pack(side="left", padx=(2, 10))
        ttk.Button(row3, text="IR を保存...", command=self.on_save).pack(
            side="left", padx=8)

        self.root.after(100, self._poll)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------- helpers
    def _browse(self, var):
        path = filedialog.askopenfilename(
            filetypes=[("WAV ファイル", "*.wav"), ("すべてのファイル", "*.*")])
        if path:
            var.set(path)

    def _on_wet(self, _=None):
        w = self.var_wet.get() / 100.0
        self.player.wet = w
        self.var_wet_lbl.set(f"{self.var_wet.get():.0f} %")

    def _set_busy(self, busy, msg=None):
        self.busy = busy
        state = "disabled" if busy else "normal"
        self.btn_make.configure(state=state)
        self.btn_play.configure(state=state)
        if msg is not None:
            self.var_status.set(msg)

    def _poll(self):
        try:
            while True:
                fn, arg = self.jobs.get_nowait()
                fn(arg)
        except queue.Empty:
            pass
        if not self.player.playing and self.btn_stop["state"] == "normal" \
                and not self.busy:
            self.btn_stop.configure(state="disabled")
            self.btn_play.configure(state="normal")
        self.root.after(100, self._poll)

    def _on_close(self):
        self.player.stop()
        self.root.destroy()

    # ------------------------------------------------------------- actions
    def on_make(self):
        rec = self.var_rec.get().strip()
        if not rec:
            messagebox.showwarning("irpond", "ターゲット録音 wav を選択してください")
            return
        src = self.var_src.get().strip() or None
        sel = self.var_len.get()
        length = None if sel == "自動" else float(sel)
        norm = self.var_norm.get()
        if src:
            msg = "測定モードで IR を計算中..."
        else:
            msg = ("推定モード(スイープ自動検出)で IR を計算中..."
                   "(数十秒かかることがあります)")
        self._set_busy(True, msg)

        def work():
            try:
                ir, sr, info = ir_core.compute_ir(
                    rec, src, length_s=length, normalize=norm)
                self.jobs.put((self._on_ir_done, (ir, sr, info)))
            except Exception as e:
                self.jobs.put((self._on_error, e))

        threading.Thread(target=work, daemon=True).start()

    def _on_ir_done(self, result):
        ir, sr, info = result
        self.ir, self.ir_sr = ir, sr
        ch = "モノラル" if ir.shape[1] == 1 else f"{ir.shape[1]}ch"
        msg = (f"IR 作成完了: {info['length_s']:.2f} 秒 / {sr} Hz / {ch}")
        if info["mode"] == "estimated":
            kind = {"exponential": "指数(Log)", "linear": "リニア"}.get(
                info.get("sweep_type"), "?")
            msg += (f"\n推定スイープ: {kind} {info['f1']:.0f} Hz → "
                    f"{info['f2']:.0f} Hz / {info['sweep_duration']:.2f} 秒 "
                    f"(フィット R²={info['fit_r2']:.4f})")
        self._set_busy(False, msg)
        self._draw_ir()

    def _on_error(self, e):
        self._set_busy(False, f"エラー: {e}")
        messagebox.showerror("irpond", str(e))

    def on_play(self):
        if self.ir is None:
            messagebox.showwarning("irpond", "先に IR を作成してください")
            return
        aud = self.var_aud.get().strip()
        if not aud:
            messagebox.showwarning("irpond", "試聴用音源 wav を選択してください")
            return
        self._set_busy(True, "試聴用音源を畳み込み中...")

        def work():
            try:
                dry, sr = ir_core.load_audio(aud)
                d, w, ch = ir_core.render_preview(dry, sr, self.ir, self.ir_sr)
                self.jobs.put((self._on_render_done,
                               (d.astype(np.float32), w.astype(np.float32), sr)))
            except Exception as e:
                self.jobs.put((self._on_error, e))

        threading.Thread(target=work, daemon=True).start()

    def _on_render_done(self, result):
        d, w, sr = result
        self._set_busy(False, "再生中(Wet スライダーはリアルタイムに効きます)")
        self._on_wet()
        try:
            self.player.play(d, w, sr)
        except Exception as e:
            self._on_error(e)
            return
        self.btn_play.configure(state="disabled")
        self.btn_stop.configure(state="normal")

    def on_stop(self):
        self.player.stop()
        self.btn_stop.configure(state="disabled")
        self.btn_play.configure(state="normal")

    def on_save(self):
        if self.ir is None:
            messagebox.showwarning("irpond", "先に IR を作成してください")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".wav", initialfile="impulse_response.wav",
            filetypes=[("WAV ファイル", "*.wav")])
        if not path:
            return
        sr_sel = self.var_sr.get()
        target_sr = None if sr_sel == "元のまま" else int(sr_sel)
        subtype = ir_core.SUBTYPES[self.var_bit.get()]
        ch_sel = self.var_ch.get()
        channels = {"元のまま": None, "モノラル": 1, "ステレオ": 2}[ch_sel]
        try:
            ir_core.save_ir(path, self.ir, self.ir_sr,
                            target_sr=target_sr, subtype=subtype,
                            channels=channels)
        except Exception as e:
            self._on_error(e)
            return
        self.var_status.set(f"保存しました: {path}")

    # ------------------------------------------------------------- drawing
    def _draw_ir(self):
        cv = self.canvas
        cv.delete("all")
        if self.ir is None:
            return
        w = max(cv.winfo_width(), 50)
        h = max(cv.winfo_height(), 40)
        n, nch = self.ir.shape
        mid = h / 2
        peak = np.abs(self.ir).max() or 1.0
        colors = ["#4da3ff", "#ff9a4d"]
        cv.create_line(0, mid, w, mid, fill="#333")
        for c in range(min(nch, 2)):
            x = self.ir[:, c] / peak
            cols = np.array_split(x, w)
            pts_hi, pts_lo = [], []
            for px, seg in enumerate(cols):
                if len(seg) == 0:
                    continue
                pts_hi.append((px, mid - seg.max() * (mid - 4)))
                pts_lo.append((px, mid - seg.min() * (mid - 4)))
            line = pts_hi + pts_lo[::-1]
            if len(line) > 2:
                cv.create_polygon(*[v for p in line for v in p],
                                  fill="", outline=colors[c])
        dur = n / self.ir_sr
        cv.create_text(6, 6, anchor="nw", fill="#aaa",
                       text=f"{dur:.2f} s / {self.ir_sr} Hz / {nch} ch",
                       font=("TkDefaultFont", 8))


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
