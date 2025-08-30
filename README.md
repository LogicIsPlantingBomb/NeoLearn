# 🧠 NeoLearn — AI Teacher with Memory

NeoLearn is an interactive **AI teacher** built with **Streamlit + LangChain + Google Gemini + Deepgram**. It supports **text and voice chat**, optional **RAG over your files** (PDF/TXT/DOCX/PPTX/Images via OCR), **TTS playback**, and rich **study tools** (notes, questions, learning reports, and export).

---
<img width="1918" height="1018" alt="Screenshot 2025-08-30 193040" src="https://github.com/user-attachments/assets/b1d39813-47f6-44db-95a3-34f5007bb901" />


## ✨ Features

* **Conversational Tutor**: Text or voice input; polite, kid-friendly explanations with a short Hinglish summary.
* **Context Memory**: Maintains chat history within the session; optionally augments with your uploaded files via FAISS.
* **RAG on Your Files**: Upload PDF/TXT/DOCX/PPTX; optional image OCR (PNG/JPG) when Tesseract is available.
* **Voice**:

  * **STT** via Deepgram (webm/opus from `streamlit_mic_recorder`).
  * **TTS** via gTTS with autoplay toggle (mute/unmute).
* **Study Tools (Sidebar)**: One‑click **Notes**, **Practice Questions**, **Learning Report**; **Personal Notes** editor.
* **Exports**: Download conversation as JSON/TXT; download generated notes, questions, and report.
* **Status & UX**: Step‑by‑step processing status, audio playback once per message, and reset controls.

---

## 🧩 Architecture (High Level)

```
Streamlit UI  ── Chat + Sidebar controls
   │
   ├─ Voice input (streamlit_mic_recorder) → Deepgram STT → transcript
   ├─ Text input → directly to LLM
   │
   ├─ File Upload → LangChain loaders → Text Splitter → Google Embeddings → FAISS
   │                                                   ↑
   ├─ Prompt (system + history + {context}) → Gemini (gemini-1.5-flash)
   │
   └─ Response → gTTS (mp3) → audio player (autoplay if unmuted)
```

---

## 🛠️ Tech Stack

* **Frontend**: Streamlit
* **LLM**: Google Gemini via `langchain_google_genai`
* **Embeddings**: Google `models/embedding-001`
* **Vector Store**: FAISS (in‑memory per session)
* **RAG**: LangChain retriever (k=3)
* **STT**: Deepgram Listen API
* **TTS**: gTTS
* **Loaders**: `PyPDFLoader`, `TextLoader`, `Docx2txtLoader`, `UnstructuredPowerPointLoader` (optional), `UnstructuredImageLoader` (optional OCR)

---

## 📦 Requirements

* **Python**: 3.9+ (3.10 recommended)
* **System (optional for OCR)**: Tesseract OCR installed and available in PATH
* **APIs**: Deepgram API key, Google API key (Gemini)

### Python packages

```bash
pip install \
  streamlit python-dotenv gTTS requests \
  langchain langchain-community langchain-google-genai langchain-text-splitters \
  faiss-cpu \
  docx2txt \
  unstructured unstructured[pdf] \
  streamlit-mic-recorder
```

> **Optional** (enable PPTX + image OCR):

```bash
pip install "unstructured[python-pptx]" pillow unstructured[pillow]
# System: install tesseract (e.g., Ubuntu: sudo apt-get install tesseract-ocr)
```

---

## 🔐 Environment Variables

Create a `.env` in project root:

```env
DEEPGRAM_API_KEY=your_deepgram_key
GOOGLE_API_KEY=your_google_api_key
```

> Get Google key from **Google AI Studio**; Deepgram from **Deepgram Console**.

---

## ▶️ Run Locally

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints.

---

## 🗂️ File Types & Loaders

* **PDF** → `PyPDFLoader`
* **TXT** → `TextLoader`
* **DOCX** → `Docx2txtLoader`
* **PPTX** → `UnstructuredPowerPointLoader` (requires `unstructured[python-pptx]`)
* **PNG/JPG** → `UnstructuredImageLoader` (requires Tesseract + `unstructured[pillow]`)

> If Tesseract is missing, the app disables image OCR and explains how to enable it.

---

## 🎛️ Usage Guide

1. **Start the app** and check the top‑left title bar; use the **mute/unmute** button to control TTS.
2. **Ask a question** via the chat input or press the **mic** to speak.
3. (Optional) **Upload files** in the sidebar before asking; the app builds a FAISS knowledge base and retrieves the top 3 chunks for context.
4. **Read or listen** to Neo’s answer; it includes a 2–3 line **Hinglish summary** and a short follow‑up question.
5. **Study Tools (Sidebar)**:

   * **Generate Notes**: structured notes from the conversation.
   * **Generate Questions**: 5–7 mixed‑type practice questions.
   * **Learning Report**: analyzes learning style, strengths, gaps, and next steps.
   * **Personal Notes**: your own notes, with download.
6. **Export** the whole chat as **JSON/TXT**.
7. **Reset** anytime with **Clear Chat & Files**.

---

## ⚙️ Configuration Details

* **Chunking**: `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)`
* **Retriever**: `k=3`
* **Gemini model**: `gemini-1.5-flash` (temperature 0.3, max tokens \~400 for answers; higher for tools)
* **Safety**: Harm blocks disabled in code for teaching breadth — adjust per your policy.
* **Audio**: gTTS MP3; autoplay only once per AI message; global mute toggle.

---

## 🧯 Troubleshooting

* **Missing keys**: App stops with a clear error if `DEEPGRAM_API_KEY` / `GOOGLE_API_KEY` are not in `.env`.
* **PPTX not loading**: Install `unstructured[python-pptx]`.
* **Image OCR disabled**: Install system **Tesseract** and Python extras `pillow` + `unstructured[pillow]`.
* **Deepgram errors**: Ensure input mime (`audio/webm;codecs=opus`) matches mic recorder output; verify key and network.
* **FAISS not found**: Ensure `faiss-cpu` is installed for your Python version.
* **Autoplay not working**: Some browsers block autoplay; click play or unmute.

---

## 🔒 Security Notes

* **API keys** must live in **server-side `.env`** only. Do **not** expose them to the client.
* Uploaded files are processed **server-side** and kept in temporary storage during indexing; purge as needed.
* Consider rate limits and safety filters for multi‑user deployments.

---

## 🚀 Deployment Options

* **Streamlit Community Cloud**: Easiest; add `.env` secrets in the project’s settings.
* **Docker**: Create a small image and pass env vars at runtime.
* **VM/Server**: `tmux`/`systemd` + reverse proxy (Caddy/NGINX) with HTTPS.

**Minimal Dockerfile** (example):

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && apt-get update && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## 🗃️ Suggested Project Structure

```
.
├── app.py                # Streamlit app (main)
├── requirements.txt      # Pinned deps
├── README.md             # This file
├── .env                  # API keys (DO NOT COMMIT)
└── data/                 # Optional: sample docs
```

**requirements.txt (example)**

```
streamlit
python-dotenv
gTTS
requests
langchain
langchain-community
langchain-google-genai
langchain-text-splitters
faiss-cpu
docx2txt
unstructured
unstructured[pdf]
unstructured[python-pptx]
streamlit-mic-recorder
pillow
```

---

## 🧭 Roadmap Ideas

* Multi-document, persistent vector stores (e.g., Chroma/Weaviate)
* Auth + per-user sessions
* Richer voice controls (pause/seek, different voices)
* Citation viewer showing which chunk answered which part
* Adjustable retrieval `k`, chunking sizes, and model selector
* Export to PDF/Docx for notes/report

---
🔑 Key Differences from ChatGPT
Personalization with Memory
ChatGPT gives generic answers, but NeoLearn remembers what you’ve uploaded and tailors responses to your study material.
RAG (Retrieval-Augmented Generation)
Instead of answering only from its training data, NeoLearn pulls directly from PDFs, PPTs, Docs, and even images you provide. This makes it more grounded and accurate for your context.
ChatGPT is mostly text-first (unless you pay for Pro), whereas NeoLearn supports voice input + voice output, plus OCR for images.
Beyond just answering questions, NeoLearn generates notes, practice questions, and learning reports — something ChatGPT doesn’t natively offer.
Unlike ChatGPT (closed system), NeoLearn is built with LangChain, FAISS, Streamlit, Gemini, Deepgram, gTTS etc., giving you full control of pipeline, embeddings, and data flow.
Data isn’t just “sent to OpenAI servers.” You decide what to feed, how it’s stored, and how it’s retrieved (vector DB).
## 📜 License

Choose a license (e.g., MIT) and add it here.

---

## 🙌 Acknowledgments

* Google AI Studio (Gemini)
* Deepgram
* LangChain & FAISS
* Streamlit community
