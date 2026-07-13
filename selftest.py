# -*- coding: utf-8 -*-
"""End-to-end self test for ir_core: synthesize sweep -> known IR -> recover."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

import ir_core

rng = np.random.default_rng(42)
FAILURES = []


def check(name, cond, detail=""):
    status = "OK  " if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        FAILURES.append(name)


def make_true_ir(sr, seconds=0.6, ch=1):
    """A plausible room/cab IR: direct spike + early taps + decaying noise."""
    n = int(sr * seconds)
    ir = np.zeros((n, ch))
    for c in range(ch):
        ir[0, c] = 1.0
        for d_ms, g in [(7, 0.5), (13, -0.35), (23, 0.25), (41, 0.15)]:
            ir[int(sr * d_ms / 1000), c] += g * (1 + 0.1 * c)
        t = np.arange(n) / sr
        tail = rng.standard_normal(n) * np.exp(-t / 0.12) * 0.1
        b, a = signal.butter(2, [80 / (sr / 2), min(0.9, 9000 / (sr / 2))],
                             "bandpass")
        ir[:, c] += signal.lfilter(b, a, tail)
    # keep the IR inside the sweep band (20 Hz - 20 kHz) so recovery is
    # physically possible; a real cab/room IR is band-limited anyway
    sos = signal.butter(8, 16000 / (sr / 2), "lowpass", output="sos")
    ir = signal.sosfiltfilt(sos, ir, axis=0)
    return ir / np.abs(ir).max()


def aligned_corr(est, ref):
    """Peak normalized cross-correlation between estimated and true IR."""
    e = est[:, 0] / (np.linalg.norm(est[:, 0]) + 1e-15)
    r = ref[:, 0] / (np.linalg.norm(ref[:, 0]) + 1e-15)
    return np.abs(signal.correlate(e, r, mode="full")).max()


def main():
    tmp = Path(tempfile.mkdtemp(prefix="irpond_test_"))
    sr = 48000

    # ---- case 1: measured mode, mono, same sr -------------------------------
    sweep = ir_core.make_ess(20, 20000, 5.0, sr)[:, None]
    true_ir = make_true_ir(sr)
    rec = signal.fftconvolve(sweep, true_ir, axes=0)
    rec = np.pad(rec, ((int(0.3 * sr), int(0.3 * sr)), (0, 0)))
    rec += rng.standard_normal(rec.shape) * 1e-5
    p_src, p_rec = tmp / "sweep.wav", tmp / "rec.wav"
    sf.write(p_src, sweep, sr, subtype="FLOAT")
    sf.write(p_rec, rec, sr, subtype="FLOAT")

    ir, out_sr, info = ir_core.compute_ir(p_rec, p_src, length_s=0.6)
    c = aligned_corr(ir, true_ir)
    check("measured mono", c > 0.99, f"corr={c:.4f} len={info['length_s']:.2f}s")

    # ---- case 2: blind mode (no source sweep) -------------------------------
    ir2, _, info2 = ir_core.compute_ir(p_rec, None, length_s=0.6)
    c2 = aligned_corr(ir2, true_ir)
    check("blind ESS", c2 > 0.85,
          f"corr={c2:.4f} type={info2.get('sweep_type')} "
          f"f={info2.get('f1', 0):.0f}->{info2.get('f2', 0):.0f}Hz "
          f"T={info2.get('sweep_duration', 0):.2f}s r2={info2.get('fit_r2', 0):.4f}")

    # ---- case 3: blind mode, linear sweep -----------------------------------
    lsweep = ir_core.make_linear_sweep(20, 20000, 5.0, sr)[:, None]
    lrec = signal.fftconvolve(lsweep, true_ir, axes=0)
    lrec = np.pad(lrec, ((int(0.2 * sr), int(0.2 * sr)), (0, 0)))
    p_lrec = tmp / "rec_lin.wav"
    sf.write(p_lrec, lrec, sr, subtype="FLOAT")
    ir3, _, info3 = ir_core.compute_ir(p_lrec, None, length_s=0.6)
    c3 = aligned_corr(ir3, true_ir)
    check("blind linear", c3 > 0.85,
          f"corr={c3:.4f} type={info3.get('sweep_type')}")

    # ---- case 4: stereo recording, mono sweep -------------------------------
    true_st = make_true_ir(sr, ch=2)
    rec_st = np.column_stack([
        signal.fftconvolve(sweep[:, 0], true_st[:, c]) for c in range(2)])
    p_rst = tmp / "rec_st.wav"
    sf.write(p_rst, rec_st, sr, subtype="FLOAT")
    ir4, _, info4 = ir_core.compute_ir(p_rst, p_src, length_s=0.6)
    c4a = aligned_corr(ir4[:, :1], true_st[:, :1])
    c4b = aligned_corr(ir4[:, 1:], true_st[:, 1:])
    check("stereo rec / mono sweep", ir4.shape[1] == 2 and min(c4a, c4b) > 0.99,
          f"ch={ir4.shape[1]} corr=({c4a:.4f},{c4b:.4f})")

    # ---- case 5: sample-rate mismatch (sweep 48k, recording 44.1k) ----------
    rec441 = ir_core.resample(rec, sr, 44100)
    rec441 *= 0.7 / np.abs(rec441).max()         # keep int format unclipped
    p_r441 = tmp / "rec441.wav"
    sf.write(p_r441, rec441, 44100, subtype="PCM_24")
    ir5, sr5, _ = ir_core.compute_ir(p_r441, p_src, length_s=0.6)
    ref441 = ir_core.resample(true_ir, sr, 44100)
    c5 = aligned_corr(ir5, ref441)
    check("sr mismatch 48k->44.1k", sr5 == 44100 and c5 > 0.98,
          f"sr={sr5} corr={c5:.4f}")

    # ---- case 6: auto length -------------------------------------------------
    ir6, _, info6 = ir_core.compute_ir(p_rec, p_src, length_s=None)
    check("auto length", 0.2 < info6["length_s"] < 2.0,
          f"len={info6['length_s']:.2f}s")

    # ---- case 7: save formats -------------------------------------------------
    for label, sub in ir_core.SUBTYPES.items():
        for tsr in (None, 96000):
            for ch in (None, 1, 2):
                p = tmp / f"ir_{sub}_{tsr}_{ch}.wav"
                ir_core.save_ir(p, ir4, out_sr, target_sr=tsr,
                                subtype=sub, channels=ch)
                info_f = sf.info(str(p))
                ok = (info_f.subtype == sub
                      and info_f.samplerate == (tsr or out_sr)
                      and info_f.channels == (ch or ir4.shape[1]))
                if not ok:
                    check(f"save {sub}/{tsr}/{ch}", False,
                          f"got {info_f.subtype}/{info_f.samplerate}/{info_f.channels}")
    check("save formats", True, "(all subtype/sr/ch combinations verified)")

    # ---- case 8: preview rendering --------------------------------------------
    tsig = rng.standard_normal((sr * 2, 1)) * 0.1
    dry, wet, ch = ir_core.render_preview(tsig, sr, ir4, out_sr)
    check("preview render", dry.shape == wet.shape and ch == 2
          and np.abs(dry).max() <= 0.91 and np.abs(wet).max() <= 0.91,
          f"shape={wet.shape} peaks=({np.abs(dry).max():.3f},{np.abs(wet).max():.3f})")

    # ---- case 9: 16-bit int input file ----------------------------------------
    p_r16 = tmp / "rec16.wav"
    sf.write(p_r16, (rec / max(np.abs(rec).max(), 1) * 0.9), sr, subtype="PCM_16")
    ir9, _, _ = ir_core.compute_ir(p_r16, p_src, length_s=0.6)
    c9 = aligned_corr(ir9, true_ir)
    check("16-bit input", c9 > 0.99, f"corr={c9:.4f}")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s): {FAILURES}")
        sys.exit(1)
    print("all tests passed")


if __name__ == "__main__":
    main()
