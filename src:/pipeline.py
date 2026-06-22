import os
import argparse
import subprocess
import pandas as pd
import torch
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline as hf_pipeline

# STEP 1: Audio extraction
def extract_audio(input_video, output_audio, ffmpeg_bin):
    os.makedirs(os.path.dirname(output_audio), exist_ok=True)
    cmd = [
        ffmpeg_bin,
        "-i", input_video,
        "-ac", "1",
        "-ar", "16000",
        output_audio,
        "-y"
    ]
    subprocess.run(cmd, check=True)
    print(f" Audio extracted to: {output_audio}")


# STEP 2: Speech-to-Text 
def transcribe_audio(audio_path, model_size="medium", device="cuda"):
    model = WhisperModel(model_size, device=device, compute_type="float16")
    segments, info = model.transcribe(audio_path, vad_filter=True, language="fa")
    results = []
    for seg in segments:
        results.append({
            "Start Time": f"{seg.start:.2f}",
            "End Time": f"{seg.end:.2f}",
            "Sentence": seg.text.strip()
        })
    df = pd.DataFrame(results)
    return df


# STEP 3: Translation
def translate_text(df, model_path, device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)

    translations = []
    for text in df["Sentence"]:
        if not isinstance(text, str) or text.strip() == "":
            translations.append("")
            continue
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128, num_beams=5)
        translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        translations.append(translated)

    df["Translation"] = translations
    return df


# STEP 4: Emotion Classification 
def classify_emotions(df):
    classifier = hf_pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", device=0 if torch.cuda.is_available() else -1)

    emotions = []
    for text in df["Translation"]:
        if not isinstance(text, str) or text.strip() == "":
            emotions.append("neutral")
            continue
        result = classifier(text, truncation=True, max_length=128)[0]
        emotions.append(result["label"])

    df["Emotion"] = emotions
    return df


# FULL PIPELINE
def full_pipeline(input_video, output_csv, model_path, ffmpeg_bin="/usr/local/lib/python3.12/dist-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"):
    audio_path = os.path.splitext(output_csv)[0] + "_audio.wav"

    # Step 1: Extract audio
    extract_audio(input_video, audio_path, ffmpeg_bin)

    # Step 2: Transcribe
    df = transcribe_audio(audio_path)

    # Step 3: Translate
    df = translate_text(df, model_path)

    # Step 4: Emotion
    df = classify_emotions(df)

    # Save final CSV
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f" Final results saved to: {output_csv}")


# ---------- MAIN ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end NLP pipeline for emotion classification")
    parser.add_argument("--input", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output", type=str, required=True, help="Path to output CSV file")
    parser.add_argument("--model", type=str, required=True, help="Path to translation model folder")
    args = parser.parse_args()

    full_pipeline(args.input, args.output, args.model)
