# predict.py
from typing import List, Optional, Tuple, Dict
from cog import BasePredictor, Input, Path

import os
import re
import io
import json
import math
import tempfile
import difflib

import numpy as np
import requests
import soundfile as sf
import librosa
import moviepy.editor as mpe


# =========================
# Whisper helper (given API)
# =========================

def extract_prompt_from_script_info(script_info: List[Tuple[str, int]]) -> str:
    """Create a prompt string for Whisper, from any script hint pieces you pass in."""
    if not script_info:
        return ""
    return ". ".join(text for text, _ in script_info)

def return_transcript(audio_path: str, script: List[Tuple[str, int]], api: str) -> dict:
    """
    Transcribes the given audio file using OpenAI's Whisper API, providing the
    script as a prompt to improve accuracy. Returns verbose JSON with word timestamps.
    """
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api}"}
    files = {
        "file": (os.path.basename(audio_path), open(audio_path, "rb"), "audio/mpeg"),
    }
    data = {
        "model": "whisper-1",
        "timestamp_granularities[]": ["word"],
        "response_format": "verbose_json",
        "prompt": extract_prompt_from_script_info(script),
    }
    resp = requests.post(url, headers=headers, files=files, data=data, timeout=300)
    resp.raise_for_status()
    return resp.json()


# =========================
# Utilities
# =========================

def normalize_token(w: str) -> str:
    # Lowercase and strip simple punctuation; keep contractions
    return re.sub(r"[^\w’'-]+", "", w.lower()).strip()

def parse_whisper_words(resp: dict) -> List[Dict]:
    """
    Returns a flat list of word dicts: {"word": str, "start": float, "end": float}
    Handles both resp["words"] and resp["segments"][i]["words"] layouts.
    """
    words = []
    if isinstance(resp, dict):
        if "words" in resp and isinstance(resp["words"], list):
            for w in resp["words"]:
                if {"word", "start", "end"} <= set(w.keys()):
                    words.append({"word": w["word"], "start": float(w["start"]), "end": float(w["end"])})
            return words
        if "segments" in resp and isinstance(resp["segments"], list):
            for seg in resp["segments"]:
                for w in seg.get("words", []) or []:
                    if {"word", "start", "end"} <= set(w.keys()):
                        words.append({"word": w["word"], "start": float(w["start"]), "end": float(w["end"])})
    return words

def build_word_lists(words: List[Dict]) -> List[str]:
    return [normalize_token(w["word"]) for w in words if normalize_token(w["word"])]

def map_indices(video_tokens: List[str], corr_tokens: List[str]) -> Dict[int, Optional[int]]:
    """
    Monotonic mapping from each VIDEO token index -> corresponding CORRECT token index.
    Robust to small diffs using difflib.
    """
    sm = difflib.SequenceMatcher(a=video_tokens, b=corr_tokens, autojunk=False)
    mapping: Dict[int, Optional[int]] = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                mapping[i1 + k] = j1 + k
        elif tag in ("replace", "delete"):
            for idx in range(i1, i2):
                mapping[idx] = None
        # 'insert' means correct has extra tokens; nothing to map from video
    return mapping

def chunk_on_pauses(words: List[Dict], pause_threshold: float = 0.35) -> List[Tuple[int, int]]:
    """
    Split word indices into chunks when the VIDEO gap between consecutive words ≥ threshold.
    Returns list of (start_idx, end_idx_exclusive).
    """
    if not words:
        return []
    chunks = []
    start = 0
    for i in range(len(words) - 1):
        gap = words[i+1]["start"] - words[i]["end"]
        if gap >= pause_threshold:
            chunks.append((start, i+1))
            start = i+1
    chunks.append((start, len(words)))
    return chunks

def load_audio_mono(path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
    y, sro = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32), sr

def slice_audio(y: np.ndarray, sr: int, start_s: float, end_s: float) -> np.ndarray:
    a = max(0, int(round(start_s * sr)))
    b = max(a, int(round(end_s * sr)))
    return y[a:b]

def add_silence(duration_s: float, sr: int) -> np.ndarray:
    n = max(0, int(round(duration_s * sr)))
    return np.zeros(n, dtype=np.float32)

def crossfade_concat(a: np.ndarray, b: np.ndarray, sr: int, xf_ms: int = 40) -> np.ndarray:
    xf = int(sr * xf_ms / 1000.0)
    if xf <= 0 or len(a) < xf or len(b) < xf:
        return np.concatenate([a, b])
    fade_out = np.linspace(1.0, 0.0, xf, endpoint=True)
    fade_in  = np.linspace(0.0, 1.0, xf, endpoint=True)
    head = a[:-xf]
    tail = a[-xf:] * fade_out + b[:xf] * fade_in
    return np.concatenate([head, tail, b[xf:]])


# =========================
# Core retiming
# =========================

def make_retimed_audio(
    video_words: List[Dict],
    corr_words: List[Dict],
    time_map: Dict[int, Optional[int]],
    corr_wave: np.ndarray,
    sr: int,
    time_stretch: bool = True,
    stretch_bounds: Tuple[float, float] = (0.8, 1.25),
    pause_threshold: float = 0.35,
    crossfade_ms: int = 40
) -> np.ndarray:
    """
    Assemble correct-voice audio following the VIDEO word/phrase timings.
    Preserves leading silence + inter-phrase pauses from the VIDEO.
    """
    if not video_words:
        return np.array([], dtype=np.float32)

    chunks = chunk_on_pauses(video_words, pause_threshold=pause_threshold)
    video_first_start = video_words[0]["start"]
    video_last_end    = video_words[-1]["end"]

    out_parts: List[np.ndarray] = []

    # Leading silence to match avatar pickup
    if video_first_start > 0:
        out_parts.append(add_silence(video_first_start, sr))

    # Helper to get correct-voice time span for a range of video indices
    def corr_bounds(v_i1: int, v_i2: int) -> Optional[Tuple[float, float]]:
        mapped = [time_map.get(i) for i in range(v_i1, v_i2) if time_map.get(i) is not None]
        if not mapped:
            return None
        c_first = min(mapped)
        c_last  = max(mapped)
        return (corr_words[c_first]["start"], corr_words[c_last]["end"])

    cursor_time = video_first_start

    for (v_i1, v_i2) in chunks:
        v_start = video_words[v_i1]["start"]
        v_end   = video_words[v_i2 - 1]["end"]
        v_dur   = max(0.0, v_end - v_start)

        # Fill any inter-chunk gap (video pauses)
        if v_start > cursor_time:
            out_parts.append(add_silence(v_start - cursor_time, sr))
            cursor_time = v_start

        cb = corr_bounds(v_i1, v_i2)
        if cb is None or v_dur <= 0:
            out_parts.append(add_silence(v_dur, sr))
            cursor_time += v_dur
            continue

        c_start, c_end = cb
        c_dur = max(0.0, c_end - c_start)
        seg = slice_audio(corr_wave, sr, c_start, c_end)

        if c_dur <= 0 or len(seg) == 0:
            out_parts.append(add_silence(v_dur, sr))
            cursor_time += v_dur
            continue

        # Stretch or pad/trim per-chunk to match video timing
        if time_stretch:
            desired_rate = max(1e-6, c_dur / max(v_dur, 1e-6))
            lo_rate = 1.0 / max(stretch_bounds[1], 1e-6)
            hi_rate = 1.0 / max(stretch_bounds[0], 1e-6)
            rate = float(np.clip(desired_rate, lo_rate, hi_rate))
            seg_ts = librosa.effects.time_stretch(seg, rate=rate)
            new_len = int(round(v_dur * sr))
            if len(seg_ts) < new_len:
                pad = add_silence((new_len - len(seg_ts)) / sr, sr)
                seg_final = crossfade_concat(seg_ts, pad, sr, crossfade_ms)
            else:
                seg_final = seg_ts[:new_len]
        else:
            new_len = int(round(v_dur * sr))
            if len(seg) < new_len:
                pad = add_silence((new_len - len(seg)) / sr, sr)
                seg_final = crossfade_concat(seg, pad, sr, crossfade_ms)
            else:
                seg_final = seg[:new_len]

        if out_parts:
            out_parts[-1] = crossfade_concat(out_parts[-1], seg_final, sr, crossfade_ms)
        else:
            out_parts.append(seg_final)

        cursor_time += v_dur

    if video_last_end > cursor_time:
        out_parts.append(add_silence(video_last_end - cursor_time, sr))

    if not out_parts:
        return np.array([], dtype=np.float32)

    y = out_parts[0]
    for i in range(1, len(out_parts)):
        y = np.concatenate([y, out_parts[i]])

    peak = np.max(np.abs(y)) if y.size else 1.0
    if peak > 0:
        y = 0.98 * (y / peak)
    return y.astype(np.float32)


# =========================
# Video/audio IO helpers
# =========================

def extract_audio_from_video(video_path: str, out_wav: str, target_sr: int = 16000) -> str:
    clip = mpe.VideoFileClip(video_path)
    tmp_wav = out_wav if out_wav.endswith(".wav") else out_wav + ".wav"
    # Write raw PCM then resample for consistent processing
    clip.audio.write_audiofile(tmp_wav, fps=48000, codec="pcm_s16le", verbose=False, logger=None)
    clip.close()
    y, sr = librosa.load(tmp_wav, sr=None, mono=True)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    sf.write(out_wav, y.astype(np.float32), sr)
    return out_wav

def mux_audio_into_video(video_in: str, wav_in: str, video_out: str) -> None:
    v = mpe.VideoFileClip(video_in)
    a = mpe.AudioFileClip(wav_in)
    v_out = v.set_audio(a)
    v_out.write_videofile(
        video_out, audio_codec="aac", codec="libx264",
        temp_audiofile=os.path.join(os.path.dirname(wav_in), "tmp-audio.m4a"),
        remove_temp=True, verbose=False, logger=None
    )
    v.close()
    a.close()
    v_out.close()


# =========================
# Cog Predictor
# =========================

class Predictor(BasePredictor):
    def setup(self) -> None:
        """
        Optional: preload anything heavy here. We keep it light; librosa/moviepy load on demand.
        """
        pass

    def predict(
        self,
        avatar_video: Path = Input(description="Talking avatar video (wrong voice)"),
        correct_voice_audio: Path = Input(description="Correct voice audio of the same script"),
        openai_api_key: str = Input(description="OpenAI API key for Whisper", default=""),
        script_hint: str = Input(
            description="(Optional) Full script text to help Whisper; improves alignment robustness",
            default=""
        ),
        time_stretch: bool = Input(description="Enable per-chunk time-stretching (pitch-preserving)", default=True),
        stretch_min_ratio: float = Input(description="Lower bound for stretch factor (new/old)", default=0.80),
        stretch_max_ratio: float = Input(description="Upper bound for stretch factor (new/old)", default=1.25),
        pause_threshold: float = Input(description="Pause split threshold (seconds)", default=0.35),
        crossfade_ms: int = Input(description="Crossfade between chunks (ms)", default=40),
        target_sr: int = Input(description="Internal processing sample rate", default=16000),
    ) -> List[Path]:
        """
        Output: a single MP4 with the avatar video revoiced using the correct voice, matched to video timing.
        """
        if not openai_api_key:
            raise ValueError("openai_api_key is required")

        script_info = [(script_hint, 0)] if script_hint else [("", 0)]

        with tempfile.TemporaryDirectory() as td:
            video_in = str(avatar_video)
            audio_in = str(correct_voice_audio)

            # 1) Extract avatar audio
            avatar_wav = os.path.join(td, "avatar.wav")
            extract_audio_from_video(video_in, avatar_wav, target_sr=target_sr)

            # 2) Ensure correct voice audio is in target_sr mono WAV
            corr_wav = os.path.join(td, "correct.wav")
            y_corr, _ = load_audio_mono(audio_in, sr=target_sr)
            sf.write(corr_wav, y_corr, target_sr)

            # 3) Transcribe both with word timestamps
            avatar_tr = return_transcript(avatar_wav, script_info, openai_api_key)
            corr_tr   = return_transcript(corr_wav, script_info, openai_api_key)

            video_words = parse_whisper_words(avatar_tr)
            corr_words  = parse_whisper_words(corr_tr)
            if not video_words:
                raise RuntimeError("No word-level timestamps found for avatar audio.")
            if not corr_words:
                raise RuntimeError("No word-level timestamps found for correct-voice audio.")

            # 4) Map words and rebuild audio to match video timing
            video_tokens = build_word_lists(video_words)
            corr_tokens  = build_word_lists(corr_words)
            idx_map = map_indices(video_tokens, corr_tokens)

            y_corr, sr = load_audio_mono(corr_wav, sr=target_sr)
            y_aligned = make_retimed_audio(
                video_words=video_words,
                corr_words=corr_words,
                time_map=idx_map,
                corr_wave=y_corr,
                sr=sr,
                time_stretch=time_stretch,
                stretch_bounds=(stretch_min_ratio, stretch_max_ratio),
                pause_threshold=pause_threshold,
                crossfade_ms=crossfade_ms,
            )

            final_wav = os.path.join(td, "aligned.wav")
            sf.write(final_wav, y_aligned, sr)

            # 5) Mux back into the original video
            out_path = os.path.join(td, "avatar_revoiced.mp4")
            mux_audio_into_video(video_in, final_wav, out_path)

            return [Path(out_path)]
