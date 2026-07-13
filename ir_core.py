# -*- coding: utf-8 -*-
"""irpond DSP core.

Impulse Response estimation from sweep measurements:
  - deconvolution of a recorded sweep against the reference sweep
  - blind mode: the reference sweep is re-synthesized from the recording
    (STFT ridge tracking + exponential / linear sweep model fit)
"""
from fractions import Fraction

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.fft import next_fast_len


# ---------------------------------------------------------------- I/O helpers

def load_audio(path):
    """Read a wav file -> (float64 array shaped (frames, channels), samplerate)."""
    data, sr = sf.read(path, always_2d=True, dtype="float64")
    if data.shape[0] == 0:
        raise ValueError(f"empty audio file: {path}")
    return data, sr


def resample(data, sr_from, sr_to):
    if sr_from == sr_to:
        return data
    frac = Fraction(int(sr_to), int(sr_from)).limit_denominator(1000)
    return signal.resample_poly(data, frac.numerator, frac.denominator, axis=0)


def to_mono(data):
    return data.mean(axis=1, keepdims=True)


def match_channels(data, ch):
    """Adapt (n, c) audio to ch channels by mixdown or duplication."""
    c = data.shape[1]
    if c == ch:
        return data
    if ch == 1:
        return to_mono(data)
    if c == 1:
        return np.repeat(data, ch, axis=1)
    return np.repeat(to_mono(data), ch, axis=1)


# --------------------------------------------------------------- sweep synth

def make_ess(f1, f2, duration, sr, fade=0.005, phase=0.0):
    """Exponential (log) sine sweep."""
    n = max(int(round(duration * sr)), 8)
    t = np.arange(n) / sr
    if abs(f2 - f1) < 1e-6 * f1:
        x = np.sin(2 * np.pi * f1 * t + phase)
    else:
        k = np.log(f2 / f1)
        x = np.sin(2 * np.pi * f1 * duration / k
                   * (np.exp(t * k / duration) - 1.0) + phase)
    return _apply_fades(x, sr, fade)


def make_linear_sweep(f1, f2, duration, sr, fade=0.005, phase=0.0):
    n = max(int(round(duration * sr)), 8)
    t = np.arange(n) / sr
    x = signal.chirp(t, f0=f1, f1=f2, t1=duration, method="linear",
                     phi=np.degrees(phase) - 90.0)      # sine convention
    return _apply_fades(x, sr, fade)


def _apply_fades(x, sr, fade):
    nf = min(int(fade * sr), len(x) // 4)
    if nf > 1:
        w = np.sin(np.linspace(0, np.pi / 2, nf)) ** 2
        x[:nf] *= w
        x[-nf:] *= w[::-1]
    return x


# ------------------------------------------------------ blind sweep recovery

def estimate_sweep(rec, sr, refine=True):
    """Estimate the reference sweep from a recorded sweep alone.

    rec: (n, ch). Returns (sweep shaped (m, 1), info dict).
    """
    mono = rec.mean(axis=1)
    t0, t1 = _active_region(mono, sr)
    if t1 - t0 < 0.05:
        raise ValueError("録音から有効なスイープ区間を検出できませんでした")

    times, freqs = _track_ridge(mono, sr, t0, t1)
    if len(times) < 8:
        raise ValueError("録音からスイープの周波数軌跡を検出できませんでした")

    # Fit exponential (ln f linear in t) and linear (f linear in t) models,
    # keep whichever explains the ridge better.
    r2_exp, coef_exp = _robust_linfit(times, np.log(freqs))
    r2_lin, coef_lin = _robust_linfit(times, freqs)

    # Anchor the sweep model at the middle of the ridge (its most reliable
    # point): f(t) = model(u_ref + rate*(t - t_ref)). The synthesized sweep
    # is trimmed to [5 Hz, Nyquist] and to the active region, so it never
    # saturates even when reverb tails inflate the detected duration.
    if r2_exp >= r2_lin:
        kind, (b0, b1) = "exponential", coef_exp
    else:
        kind, (b0, b1) = "linear", coef_lin
    t_ref = float(np.median(times))
    u_ref = b0 + b1 * t_ref
    rate = float(b1)
    if abs(rate) < 1e-9:
        raise ValueError("スイープらしい周波数変化が検出できませんでした")

    built = _build_sweep(kind, rate, t0, t1, t_ref, u_ref, sr)
    if built is None:
        raise ValueError("スイープらしい周波数変化が検出できませんでした")

    # The ridge fit is only good to ~0.1-1% in sweep rate, but a rate error
    # of eps smears the IR over eps*duration seconds. Refine the rate by
    # maximizing the crest factor of the deconvolution.
    if refine:
        rate = _refine_rate(mono, sr, kind, rate, t0, t1, t_ref, u_ref)
        phase = _fit_phase(mono, sr, kind, rate, t0, t1, t_ref, u_ref)
        built = _build_sweep(kind, rate, t0, t1, t_ref, u_ref, sr,
                             phase=phase) or built
    sweep, f1, f2, duration = built

    info = {"sweep_type": kind, "f1": f1, "f2": f2, "sweep_duration": duration,
            "fit_r2": float(max(r2_exp, r2_lin))}
    return sweep[:, None], info


def _build_sweep(kind, rate, t0, t1, t_ref, u_ref, sr, fmin=5.0, phase=0.0):
    """Synthesize the model sweep, trimmed to [fmin, Nyquist] x [t0, t1].

    Returns (sweep, f1, f2, duration) or None if the span degenerates.
    """
    nyq = 0.499 * sr
    if kind == "exponential":
        freq_at = lambda t: float(np.exp(u_ref + rate * (t - t_ref)))
        time_at = lambda f: t_ref + (np.log(f) - u_ref) / rate
    else:
        freq_at = lambda t: float(u_ref + rate * (t - t_ref))
        time_at = lambda f: t_ref + (f - u_ref) / rate
    ta, tb = sorted((time_at(fmin), time_at(nyq)))
    ts, te = max(t0, ta), min(t1, tb)
    duration = te - ts
    if duration < 0.05:
        return None
    f1, f2 = freq_at(ts), freq_at(te)
    if abs(f2 - f1) < 1.0:
        return None
    if kind == "exponential":
        sweep = make_ess(f1, f2, duration, sr, phase=phase)
    else:
        sweep = make_linear_sweep(f1, f2, duration, sr, phase=phase)
    return sweep, f1, f2, duration


def _refine_rate(mono, sr, kind, rate, t0, t1, t_ref, u_ref, span=0.02):
    """Refine the sweep rate by maximizing deconvolution crest factor.

    A rate error of eps smears the IR over ~eps*duration, so the crest
    factor (peak/RMS) of the deconvolved buffer peaks extremely sharply at
    the true rate (basin width ~0.05% of the rate). Scan densely enough to
    land inside that basin, then zoom in.
    """
    nfft = next_fast_len(len(mono) + int((t1 - t0) * sr) + 16)
    R = np.fft.rfft(mono, nfft)

    def crest(r, env=False):
        built = _build_sweep(kind, r, t0, t1, t_ref, u_ref, sr)
        if built is None:
            return -1.0
        S = np.fft.rfft(built[0], nfft)
        P = (S * S.conj()).real
        H = R * np.conj(S) / (P + 1e-6 * P.max())
        if env:
            # analytic-signal envelope: immune to sub-sample scalloping
            e = np.abs(_analytic(H, nfft))
            return e.max() / (np.sqrt((e ** 2).mean()) + 1e-15)
        h = np.fft.irfft(H, nfft)
        return np.abs(h).max() / (np.sqrt((h ** 2).mean()) + 1e-15)

    # dense scan at 1/4 basin width, then two envelope-based zoom rounds
    step = 2.5e-4 * abs(rate)
    xs = rate + np.arange(-span * abs(rate), span * abs(rate) + step, step)
    vals = [crest(x) for x in xs]
    best_i = int(np.argmax(vals))
    # if the peak sits at the scan edge, the initial estimate was off by
    # more than `span` -- keep extending in that direction
    for _ in range(5):
        if 0 < best_i < len(xs) - 1:
            break
        direction = -1 if best_i == 0 else 1
        ext = xs[best_i] + direction * step * np.arange(1, int(
            span * abs(rate) / step) + 1)
        ext_vals = [crest(x) for x in ext]
        if direction == -1:
            xs = np.concatenate([ext[::-1], xs])
            vals = ext_vals[::-1] + vals
        else:
            xs = np.concatenate([xs, ext])
            vals = vals + ext_vals
        best_i = int(np.argmax(vals))
    best = float(xs[best_i])
    for _ in range(2):
        xs = np.linspace(best - step, best + step, 21)
        best = float(xs[int(np.argmax([crest(x, env=True) for x in xs]))])
        step = step / 10.0
    return best


def _analytic(H_rfft, nfft):
    """Analytic signal from a one-sided (rfft) spectrum."""
    full = np.zeros(nfft, dtype=complex)
    full[0] = H_rfft[0]
    if nfft % 2 == 0:
        full[1:nfft // 2] = 2.0 * H_rfft[1:-1]
        full[nfft // 2] = H_rfft[-1]
    else:
        full[1:(nfft + 1) // 2] = 2.0 * H_rfft[1:]
    return np.fft.ifft(full)


def _fit_phase(mono, sr, kind, rate, t0, t1, t_ref, u_ref, iters=3):
    """Estimate the start-phase offset between the real and synthetic sweep.

    A phase mismatch dphi rotates the deconvolved IR into
    cos(dphi)*h - sin(dphi)*hilbert(h); the analytic-signal phase at the
    envelope peak measures it directly.
    """
    nfft = next_fast_len(len(mono) + int((t1 - t0) * sr) + 16)
    R = np.fft.rfft(mono, nfft)
    phase = 0.0
    for _ in range(iters):
        built = _build_sweep(kind, rate, t0, t1, t_ref, u_ref, sr,
                             phase=phase)
        if built is None:
            return 0.0
        S = np.fft.rfft(built[0], nfft)
        P = (S * S.conj()).real
        z = _analytic(R * np.conj(S) / (P + 1e-6 * P.max()), nfft)
        dphi = float(np.angle(z[int(np.abs(z).argmax())]))
        phase += dphi
        if abs(dphi) < 0.02:
            break
    return phase


def _active_region(mono, sr, thresh_db=-35.0):
    """(t_start, t_end) of the sweep inside the recording, via RMS envelope."""
    win = max(int(0.010 * sr), 16)
    hop = win // 2
    n = (len(mono) - win) // hop + 1
    if n < 2:
        return 0.0, len(mono) / sr
    idx = np.arange(n) * hop
    frames = np.lib.stride_tricks.sliding_window_view(mono, win)[::hop][:n]
    rms = np.sqrt((frames ** 2).mean(axis=1)) + 1e-12
    db = 20 * np.log10(rms)
    above = np.flatnonzero(db > db.max() + thresh_db)
    if len(above) == 0:
        return 0.0, len(mono) / sr
    t_start = idx[above[0]] / sr
    t_end = (idx[above[-1]] + win) / sr
    return t_start, t_end


def _track_ridge(mono, sr, t0, t1, keep_db=-40.0):
    """Instantaneous-frequency ridge of the sweep via STFT peak tracking."""
    nper = 4096 if sr >= 32000 else 2048
    nper = min(nper, max(256, len(mono) // 8))
    f, t, Z = signal.stft(mono, fs=sr, nperseg=nper, noverlap=nper * 3 // 4,
                          padded=False, boundary=None)
    mag = np.abs(Z)
    if mag.shape[1] < 4:
        return np.array([]), np.array([])
    peak_bin = mag.argmax(axis=0)
    peak_mag = mag.max(axis=0)

    # parabolic interpolation around the peak bin for sub-bin accuracy
    freqs = np.empty(mag.shape[1])
    df = f[1] - f[0]
    for i in range(mag.shape[1]):
        b = peak_bin[i]
        if 0 < b < mag.shape[0] - 1:
            a, c, d = mag[b - 1, i], mag[b, i], mag[b + 1, i]
            denom = a - 2 * c + d
            delta = 0.5 * (a - d) / denom if abs(denom) > 1e-30 else 0.0
            freqs[i] = (b + np.clip(delta, -0.5, 0.5)) * df
        else:
            freqs[i] = b * df

    keep = peak_mag > peak_mag.max() * 10 ** (keep_db / 20)
    keep &= (t >= t0) & (t <= t1)
    keep &= freqs > df * 0.5
    times, freqs = t[keep], freqs[keep]
    if len(times) > 12:                     # drop edge frames (fades, onset)
        times, freqs = times[2:-2], freqs[2:-2]
    return times, freqs


def _robust_linfit(x, y, passes=3):
    """Linear fit with iterative sigma clipping. Returns (r2, (b0, b1)).

    Reverb tails ringing past the sweep end put a correlated cluster of
    off-model frames at the edges of the ridge; repeated clipping at 2 sigma
    removes them where a single loose pass would not.
    """
    def fit(x, y):
        b1, b0 = np.polyfit(x, y, 1)
        resid = y - (b0 + b1 * x)
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1.0 - (resid ** 2).sum() / ss_tot if ss_tot > 0 else 0.0
        return r2, (b0, b1), resid

    r2, coef, resid = fit(x, y)
    for _ in range(passes):
        s = resid.std()
        if s <= 0:
            break
        keep = np.abs(resid) < 2.0 * s
        if keep.sum() < max(8, len(x) // 3) or keep.all():
            break
        x, y = x[keep], y[keep]
        r2, coef, resid = fit(x, y)
    return r2, coef


# -------------------------------------------------------------- deconvolution

def deconvolve(rec, src, reg=1e-6):
    """Regularized spectral division. rec (n, cr), src (m, cs) at the same sr.

    Returns the full circular IR (nfft, cr).
    """
    n, cr = rec.shape
    m, cs = src.shape
    if cs != cr and cs != 1:
        src = to_mono(src)
        cs = 1
    nfft = next_fast_len(n + m)
    out = np.empty((nfft, cr))
    spec_cache = {}
    for ri in range(cr):
        si = ri if cs > 1 else 0
        if si not in spec_cache:
            S = np.fft.rfft(src[:, si], nfft)
            P = np.abs(S) ** 2
            spec_cache[si] = (np.conj(S), P, reg * P.max())
        Sc, P, eps = spec_cache[si]
        R = np.fft.rfft(rec[:, ri], nfft)
        out[:, ri] = np.fft.irfft(R * Sc / (P + eps), nfft)
    return out


def extract_ir(ir_full, sr, length_s=None, normalize=True, pre_ms=2.0):
    """Cut the causal IR out of the circular deconvolution result."""
    absmax = np.abs(ir_full).max(axis=1)
    peak = int(absmax.argmax())
    pre = int(sr * pre_ms / 1000)
    rolled = np.roll(ir_full, pre - peak, axis=0)

    if length_s is None:
        length = _auto_length(rolled, sr)
    else:
        length = max(int(sr * length_s), 16)
    length = min(length, rolled.shape[0])
    ir = rolled[:length].copy()

    # short fade-in over the pre-roll, half-Hann fade-out on the tail
    if pre > 2:
        nf = pre // 2
        ir[:nf] *= np.sin(np.linspace(0, np.pi / 2, nf))[:, None] ** 2
    nf = max(min(length // 10, int(0.05 * sr)), 8)
    nf = min(nf, length)
    ir[-nf:] *= (np.cos(np.linspace(0, np.pi / 2, nf)) ** 2)[:, None]

    if normalize:
        p = np.abs(ir).max()
        if p > 0:
            ir *= 10 ** (-1.0 / 20) / p          # peak at -1 dBFS
    return ir


def _auto_length(rolled, sr):
    """Find where the IR decays into the noise floor."""
    cap = min(int(10 * sr), rolled.shape[0] // 2)
    seg = np.abs(rolled[:cap]).max(axis=1)
    win = max(int(0.005 * sr), 8)
    n = len(seg) // win
    if n < 4:
        return cap
    env = seg[:n * win].reshape(n, win).max(axis=1) + 1e-15
    db = 20 * np.log10(env)
    noise = np.median(db[int(n * 0.7):])
    thresh = max(db.max() - 60.0, noise + 10.0)
    above = np.flatnonzero(db > thresh)
    end = (above[-1] + 1) * win if len(above) else cap
    end = int(end * 1.2) + int(0.01 * sr)        # 20% margin
    return int(np.clip(end, int(0.05 * sr), cap))


# ------------------------------------------------------------------ pipeline

def compute_ir(rec_path, src_path=None, length_s=None, normalize=True, reg=1e-6):
    """Full pipeline. Returns (ir (n, ch), sr, info dict)."""
    rec, sr = load_audio(rec_path)
    info = {}
    if src_path:
        src, sr_src = load_audio(src_path)
        if sr_src != sr:
            src = resample(src, sr_src, sr)
        info["mode"] = "measured"
    else:
        src, sweep_info = estimate_sweep(rec, sr)
        info.update(sweep_info)
        info["mode"] = "estimated"
    full = deconvolve(rec, src, reg=reg)
    ir = extract_ir(full, sr, length_s=length_s, normalize=normalize)
    info.update(sr=sr, channels=ir.shape[1], length_s=ir.shape[0] / sr)
    return ir, sr, info


# ------------------------------------------------------------------- preview

def render_preview(dry, sr, ir, ir_sr):
    """Convolve audition audio with the IR.

    Returns (dry_padded, wet, channels) — equal-length float arrays scaled so
    that any linear dry/wet mix stays below full scale.
    """
    ir = resample(ir, ir_sr, sr)
    ch = min(max(dry.shape[1], ir.shape[1]), 2)
    d = match_channels(dry, ch)
    h = match_channels(ir, ch)
    wet = np.stack(
        [signal.fftconvolve(d[:, c], h[:, c]) for c in range(ch)], axis=1)
    dpad = np.zeros_like(wet)
    dpad[:d.shape[0]] = d

    rms_d = np.sqrt((dpad ** 2).mean())
    rms_w = np.sqrt((wet ** 2).mean())
    if rms_w > 1e-12:
        wet *= rms_d / rms_w                     # loudness-match wet to dry
    scale = 0.9 / max(np.abs(dpad).max(), np.abs(wet).max(), 1e-9)
    return dpad * scale, wet * scale, ch


# ---------------------------------------------------------------------- save

SUBTYPES = {"16-bit PCM": "PCM_16", "24-bit PCM": "PCM_24",
            "32-bit float": "FLOAT"}


def save_ir(path, ir, sr, target_sr=None, subtype="PCM_24", channels=None):
    data = ir
    if channels:
        data = match_channels(data, channels)
    if target_sr and target_sr != sr:
        data = resample(data, sr, target_sr)
        sr = int(target_sr)
    if subtype != "FLOAT":
        peak = np.abs(data).max()
        if peak > 0.999:
            data = data * (0.999 / peak)
    sf.write(path, data, sr, subtype=subtype)
