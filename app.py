import os
import json
import faiss
import numpy as np
import tempfile
import streamlit as st
import torch
import soundfile as sf
import scipy.signal
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from langdetect import detect
import whisper
from audio_recorder_streamlit import audio_recorder

# Initialize OpenAI and Whisper
client = OpenAI()
stt = whisper.load_model("base", device="cpu")
model = SentenceTransformer("all-MiniLM-L6-v2")
model.to("cpu")

# Load and chunk company data
with open('arslanasghar_full_content.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()

chunk_size = 500
chunks = [raw_text[i:i + chunk_size] for i in range(0, len(raw_text), chunk_size)]
embeddings = model.encode(chunks)
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

# Create memory and log folders
os.makedirs("memory", exist_ok=True)
os.makedirs("logs", exist_ok=True)

def retrieve_context(query, threshold=0.5, top_k=3):
    query_emb = model.encode([query])
    scores, indices = index.search(query_emb, top_k)
    if scores[0][0] >= threshold:
        return [chunks[i] for i in indices[0]]
    return []

def load_memory(user_id):
    path = f"memory/user_{user_id}.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_memory(user_id, messages):
    path = f"memory/user_{user_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

def append_log(user_id, user_text, ai_reply, intent=None):
    log_path = f"logs/{user_id}.log"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"User: {user_text}\nIntent: {intent}\nArslan: {ai_reply}\n{'-'*50}\n")

def detect_intent(user_text):
    keywords = {
        "web": ["website", "web development", "site"],
        "seo": ["seo", "search engine"],
        "ads": ["ad", "google ads", "facebook ads", "campaign"],
        "pricing": ["price", "cost", "rate"],
        "support": ["help", "support", "issue"],
    }
    for intent, keys in keywords.items():
        if any(word in user_text.lower() for word in keys):
            return intent
    return "general"

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Arslan - Voice Assistant", layout="centered")
st.title("🎙️ Arslan — Your Human Digital Consultant")
st.markdown("Speak with your mic or upload a voice message — Arslan understands, remembers, and replies.")

user_id = st.text_input("🔐 Enter your User ID:")
mic_audio = audio_recorder(text="🎙️ Click to record", pause_threshold=3.0, sample_rate=16000)
file_audio = st.file_uploader("📂 Or upload a WAV audio file:", type=["wav"])

typed_text = None
if user_id and not mic_audio and not file_audio:
    typed_text = st.text_input("📝 Or type your question:")

if user_id and (mic_audio or file_audio or typed_text):
    if mic_audio:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(mic_audio)
            audio_path = tmp.name
    elif file_audio:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(file_audio.read())
            audio_path = tmp.name
    else:
        audio_path = None

    if audio_path:
        # Load WAV using soundfile
        audio_array, sample_rate = sf.read(audio_path)

        # Convert to mono if stereo
        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=1)

        # Resample to 16000 Hz
        if sample_rate != 16000:
            num_samples = int(len(audio_array) * 16000 / sample_rate)
            audio_array = scipy.signal.resample(audio_array, num_samples)
            sample_rate = 16000

        # Normalize and convert to tensor
        audio_array = audio_array / np.max(np.abs(audio_array))
        audio_tensor = torch.tensor(audio_array).float()
        audio_tensor = whisper.pad_or_trim(audio_tensor)

        mel = whisper.log_mel_spectrogram(audio_tensor).to(stt.device)
        options = whisper.DecodingOptions(fp16=False)
        result = whisper.decode(stt, mel, options)
        user_text = result.text.strip()

        if not user_text:
            st.warning("Sorry, couldn't understand the audio. Try again.")
            st.stop()
    else:
        user_text = typed_text.strip()

    language = detect(user_text)
    memory = load_memory(user_id)
    context = retrieve_context(user_text)
    context_text = "\n".join(context) if context else ""
    intent = detect_intent(user_text)

    discount_offer = ""
    if intent == "pricing":
        discount_offer = "You're eligible for a special 10% discount on all services today!"

    if not memory:
        memory.append({"role": "assistant", "content": "Hi there! I'm Arslan. May I know your name?"})
    elif len(memory) == 1:
        memory.append({"role": "assistant", "content": "Nice to meet you! How can I help you today?"})

    memory.append({"role": "user", "content": user_text})

    system_prompt = (
        f"You are Arslan, a friendly and human digital marketer from Doha, Qatar. "
        f"You are not an AI. Keep responses short and professional. Act like a real person and base your answers on the provided data. "
        f"Respond in {language.upper()} language."
    )
    if context_text:
        system_prompt += f"\nRelevant info: {context_text}"
    if discount_offer:
        memory.append({"role": "assistant", "content": discount_offer})

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": system_prompt}] + memory[-10:]
    )
    ai_reply = response.choices[0].message.content.strip()

    memory.append({"role": "assistant", "content": ai_reply})
    save_memory(user_id, memory)
    append_log(user_id, user_text, ai_reply, intent)

    st.markdown(f"**🗣️ You said:** {user_text}")
    st.markdown(f"**🤖 Arslan replied:** {ai_reply}")

    with open("response.mp3", "wb") as f:
        audio_data = client.audio.speech.create(
            model="tts-1",
            voice="echo",
            input=ai_reply
        )
        f.write(audio_data.content)

    st.audio("response.mp3", format="audio/mp3")

elif user_id:
    st.info("Please record, upload, or type your message to start.")
