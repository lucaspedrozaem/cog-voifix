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
import shutil

import numpy as np
import requests
import soundfile as sf
import librosa
import pyrubberband as pyrb
import moviepy.editor as mpe


# =========================
# Whisper helper
# =========================

def extract_prompt_from_script_info(script_info: List[Tuple[str, int]]) -> str:
    if not script_info:
        return ""
    return ". ".join(text for text, _ in script_info)

def return_transcript(audio_path: str, script: List[Tuple[str, int]], api: str) -> dict:
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
    return re.sub(r"[^\w’'-]+", "", w.lower()).strip()

def parse_whisper_words(resp: dict) -> List[Dict]:
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
    sm = difflib.SequenceMatcher(a=video_tokens, b=corr_tokens, autojunk=False)
    mapping: Dict[int, Optional[int]] = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                mapping[i1 + k] = j1 + k
        elif tag in ("replace", "delete"):
            for idx in range(i1, i2):
                mapping[idx] = None
    return mapping

def chunk_on_pauses(words: List[Dict], pause_threshold: float = 0.35) -> List[Tuple[int, int]]:
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
# Core retiming (Rubberband)
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
    if not video_words:
        return np.array([], dtype=np.float32)

    chunks = chunk_on_pauses(video_words, pause_threshold=pause_threshold)
    video_last_end    = video_words[-1]["end"]
    out_parts: List[np.ndarray] = []
    cursor_time = 0.0

    def corr_bounds(v_i1: int, v_i2: int) -> Optional[Tuple[float, float]]:
        mapped = [time_map.get(i) for i in range(v_i1, v_i2) if time_map.get(i) is not None]
        if not mapped:
            return None
        c_first = min(mapped)
        c_last  = max(mapped)
        return (corr_words[c_first]["start"], corr_words[c_last]["end"])

    for (v_i1, v_i2) in chunks:
        v_start = video_words[v_i1]["start"]
        v_end   = video_words[v_i2 - 1]["end"]
        v_dur   = max(0.0, v_end - v_start)

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

        if time_stretch:
            desired_rate = max(1e-6, c_dur / max(v_dur, 1e-6))
            lo_rate = 1.0 / max(stretch_bounds[1], 1e-6)
            hi_rate = 1.0 / max(stretch_bounds[0], 1e-6)
            rate = float(np.clip(desired_rate, lo_rate, hi_rate))
            
            try:
                seg_ts = pyrb.time_stretch(seg, sr, rate=rate)
            except Exception as e:
                print(f"WARNING: Rubberband failed: {e}. Falling back to simple padding.")
                seg_ts = seg

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
    print(f"   Extracting audio from {video_path} to {out_wav}...")
    clip = mpe.VideoFileClip(video_path)
    tmp_wav = out_wav if out_wav.endswith(".wav") else out_wav + ".wav"
    clip.audio.write_audiofile(tmp_wav, fps=48000, codec="pcm_s16le", verbose=False, logger=None)
    clip.close()
    
    y, sr = librosa.load(tmp_wav, sr=None, mono=True)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    sf.write(out_wav, y.astype(np.float32), sr)
    return out_wav

def mux_audio_into_video(video_in: str, wav_in: str, video_out: str) -> None:
    print(f"   Muxing {wav_in} into {video_in} -> {video_out}...")
    v = mpe.VideoFileClip(video_in)
    a = mpe.AudioFileClip(wav_in)
    
    if abs(a.duration - v.duration) > 0.5:
         print(f"WARNING: Final audio duration ({a.duration:.2f}s) differs from video ({v.duration:.2f}s).")

    v_out = v.set_audio(a)
    v_out.write_videofile(
        video_out, audio_codec="aac", codec="libx264",
        temp_audiofile=os.path.join(os.path.dirname(wav_in), "tmp-audio.m4a"),
        remove_temp=True, verbose=False, logger=None
    )
    v.close()
    a.close()
    v_out.close()
    print("   Muxing complete.")


# =========================
# Cog Predictor
# =========================

class Predictor(BasePredictor):
    def setup(self) -> None:
        print("✅ Predictor setup complete.")

    def predict(
        self,
        avatar_video: Path = Input(description="Talking avatar video (wrong voice)"),
        correct_voice_audio: Path = Input(description="Correct voice audio of the same script"),
        openai_api_key: str = Input(description="OpenAI API key for Whisper", default=""),
        script_hint: str = Input(default=""),
        time_stretch: bool = Input(default=True),
        audio_start_offset_ms: int = Input(
            description="Manual offset override in ms. If 0, auto-calibration is attempted.", 
            default=0
        ),
        stretch_min_ratio: float = Input(default=0.80),
        stretch_max_ratio: float = Input(default=1.25),
        pause_threshold: float = Input(default=0.35),
        crossfade_ms: int = Input(default=40),
        target_sr: int = Input(default=16000),
    ) -> List[Path]:
        print("🚀 Prediction started...")
        if not openai_api_key:
            raise ValueError("openai_api_key is required")

        script_info = [(script_hint, 0)] if script_hint else [("", 0)]
        temp_dir: Optional[str] = None
        
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                out_path = f.name
            print(f"   Set final output path to: {out_path}")

            temp_dir = tempfile.mkdtemp()
            video_in = str(avatar_video)
            audio_in = str(correct_voice_audio)

            print("✅ 1. Extracting/Preparing audio...")
            avatar_wav_path = os.path.join(temp_dir, "avatar.wav")
            extract_audio_from_video(video_in, avatar_wav_path, target_sr=target_sr)
            corr_wav_path = os.path.join(temp_dir, "correct.wav")
            y_corr_orig, _ = load_audio_mono(audio_in, sr=target_sr)
            sf.write(corr_wav_path, y_corr_orig, target_sr)

            print("✅ 2. Transcribing...")
            avatar_tr = return_transcript(avatar_wav_path, script_info, openai_api_key)
            corr_tr   = return_transcript(corr_wav_path, script_info, openai_api_key)
            video_words = parse_whisper_words(avatar_tr)
            corr_words  = parse_whisper_words(corr_tr)
            if not video_words or not corr_words: raise RuntimeError("Failed to find words in one of the inputs.")

            # ================================================================
            # +++ AUTO-CALIBRATION LOGIC +++
            # ================================================================
            whisper_start = video_words[0]['start']
            
            # Only run auto-calibration if:
            # 1. No manual offset was provided (offset == 0)
            # 2. Whisper thinks it starts almost instantly (< 50ms)
            if audio_start_offset_ms == 0 and whisper_start < 0.05:
                print("🕵️‍♂️ Auto-calibrating start time (Whisper reported ~0s)...")
                y_avatar, sr_avatar = load_audio_mono(avatar_wav_path, sr=target_sr)
                
                # Use the magical delta=0.2 that worked in your logs
                onsets = librosa.onset.onset_detect(y=y_avatar, sr=sr_avatar, units='time', delta=0.2)
                
                if len(onsets) > 0:
                    first_onset = onsets[0]
                    # If we found a start time that is significantly later than Whisper's 0.0s
                    if first_onset > 0.1: 
                         print(f"🎯 Auto-calibration found true start at {first_onset:.3f}s. Overriding Whisper.")
                         # This single line fixes everything downstream. 
                         # The retimer will now automatically insert this much silence at the start.
                         video_words[0]['start'] = first_onset
                    else:
                         print(f"   Auto-calibration confirmed start near 0s ({first_onset:.3f}s).")
                else:
                    print("⚠️ Auto-calibration failed to detect any strong onsets. Using Whisper default.")
            # ================================================================

            print("✅ 3. Aligning and retiming...")
            idx_map = map_indices(build_word_lists(video_words), build_word_lists(corr_words))

            y_corr, sr = load_audio_mono(corr_wav_path, sr=target_sr)
            y_aligned = make_retimed_audio(
                video_words, corr_words, idx_map, y_corr, sr,
                time_stretch, (stretch_min_ratio, stretch_max_ratio),
                pause_threshold, crossfade_ms
            )

            # Apply manual offset if provided (overrides auto-calibration because of the if check above)
            if audio_start_offset_ms != 0:
                print(f"👉 Applying manual offset: {audio_start_offset_ms}ms")
                offset_samples = int(round(audio_start_offset_ms * sr / 1000.0))
                if offset_samples > 0:
                     y_aligned = np.concatenate([np.zeros(offset_samples, dtype=np.float32), y_aligned])
                elif offset_samples < 0:
                     trim_idx = abs(offset_samples)
                     y_aligned = y_aligned[trim_idx:] if trim_idx < len(y_aligned) else np.array([], dtype=np.float32)

            if y_aligned.size == 0 or np.max(np.abs(y_aligned)) < 1e-4:
                 raise RuntimeError("Processing produced empty or silent audio.")

            final_wav_path = os.path.join(temp_dir, "aligned.wav")
            sf.write(final_wav_path, y_aligned, sr)

            print("✅ 4. Muxing final video...")
            mux_audio_into_video(video_in, final_wav_path, out_path)

            if not os.path.exists(out_path) or os.path.getsize(out_path) < 1000:
                 raise RuntimeError("Final video file is missing or too small.")

            print("🏁 Success.")
            return [Path(out_path)]

        finally:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)