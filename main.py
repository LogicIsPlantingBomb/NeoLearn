import os
import time
import tempfile
import requests
import json
from datetime import datetime
from io import BytesIO
from dotenv import load_dotenv
from gtts import gTTS
import streamlit as st

# LangChain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings, HarmBlockThreshold, HarmCategory
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, SystemMessage

# For PPTX
try:
    from langchain_community.document_loaders import UnstructuredPowerPointLoader
except ImportError:
    pass

# For OCR (images)
try:
    from langchain_community.document_loaders import UnstructuredImageLoader
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

# -----------------------------
# 2. Load Environment Variables
# -----------------------------
load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not DEEPGRAM_API_KEY:
    st.error("❌ `DEEPGRAM_API_KEY` not found in `.env`!")
    st.stop()
if not GOOGLE_API_KEY:
    st.error("❌ `GOOGLE_API_KEY` not found in `.env`!")
    st.info("Get it from: https://aistudio.google.com/app/apikey")
    st.stop()

# -----------------------------
# 3. Initialize Session State
# -----------------------------
if 'messages' not in st.session_state:
    st.session_state.messages = []  # Chat history: list of dict {role, content, type, file_name}

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

if 'status' not in st.session_state:
    st.session_state.status = "Ready"

if 'transcript' not in st.session_state:
    st.session_state.transcript = ""

if 'audio_bytes' not in st.session_state:
    st.session_state.audio_bytes = None

if 'audio_muted' not in st.session_state:
    st.session_state.audio_muted = False

if 'processing_status' not in st.session_state:
    st.session_state.processing_status = []

# Track played audio to avoid replay
if 'played_audio_keys' not in st.session_state:
    st.session_state.played_audio_keys = set()

# -----------------------------
# 4. Page Config
# -----------------------------
st.set_page_config(page_title="🧠 NeoLearn - AI Assistant with Memory", layout="centered")

# Header with title and controls
col1, col2 = st.columns([4, 1])
with col1:
    st.title("🧠 NeoLearn")
    st.caption("Your AI Teacher Neo - Learning Made Interactive")

with col2:
    # Mute button
    if st.button("🔊" if st.session_state.audio_muted else "🔇", help="Toggle audio mute"):
        st.session_state.audio_muted = not st.session_state.audio_muted

st.markdown("""
Ask a question **with or without uploading** a file (PDF).
Neo remembers the conversation and answers contextually.
""")

# Status display
if st.session_state.processing_status:
    status_container = st.container()
    with status_container:
        st.markdown("**Processing Status:**")
        for status in st.session_state.processing_status:
            if status.get("completed"):
                st.success(f"✅ {status['message']}")
            else:
                st.info(f"⏳ {status['message']}")

# -----------------------------
# 5. File Upload (Multiple Types)
# -----------------------------")
st.sidebar.header("📁 Upload Reference File (Optional)")
# Only allow file types that don't require OCR if tesseract is not available
allowed_types = ["pdf", "txt", "docx", "pptx"]
if HAS_OCR:
    try:
        # Test if tesseract is actually available
        import subprocess
        subprocess.run(['tesseract', '--version'], capture_output=True, check=True)
        allowed_types.extend(["png", "jpg", "jpeg"])
        upload_help = "Upload PDF, TXT, DOCX, PPTX, or Image files"
    except (subprocess.CalledProcessError, FileNotFoundError):
        upload_help = "Upload PDF, TXT, DOCX, or PPTX files (Image support disabled - Tesseract not found)"
else:
    upload_help = "Upload PDF, TXT, DOCX, or PPTX files (Image support not installed)"

uploaded_file = st.sidebar.file_uploader(
    upload_help,
    type=allowed_types,
    key="file_uploader"
)

# Helper function to update status
def update_status(message, completed=False):
    st.session_state.processing_status.append({
        "message": message,
        "completed": completed,
        "timestamp": datetime.now()
    })
    st.rerun()

# Process uploaded file
if uploaded_file:
    file_ext = uploaded_file.name.split(".")[-1].lower()
    file_key = f"{uploaded_file.name}_{uploaded_file.size}"

    # Avoid reprocessing same file
    if 'last_file_key' not in st.session_state or st.session_state.last_file_key != file_key:
        st.session_state.last_file_key = file_key
        st.session_state.vector_store = None  # Reset
        st.session_state.processing_status = []  # Clear previous status
        
        with st.sidebar:
            with st.spinner(f"🧠 Processing {uploaded_file.name}..."):
                # Update status: Starting
                st.session_state.processing_status.append({
                    "message": f"Starting to process {uploaded_file.name}",
                    "completed": False
                })

                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                try:
                    # Update status: Loading file
                    st.session_state.processing_status.append({
                        "message": f"Loading {file_ext.upper()} file",
                        "completed": False
                    })

                    # Load based on type
                    if file_ext == "pdf":
                        loader = PyPDFLoader(tmp_path)
                    elif file_ext == "txt":
                        loader = TextLoader(tmp_path, encoding='utf-8')
                    elif file_ext == "docx":
                        loader = Docx2txtLoader(tmp_path)
                    elif file_ext == "pptx":
                        if 'UnstructuredPowerPointLoader' in globals():
                            loader = UnstructuredPowerPointLoader(tmp_path)
                        else:
                            st.sidebar.error("Install: `pip install unstructured-python-pptx`")
                            raise Exception("PPTX support missing")
                    elif file_ext in ["png", "jpg", "jpeg"]:
                        # Check if tesseract is available
                        try:
                            import subprocess
                            subprocess.run(['tesseract', '--version'], capture_output=True, check=True)
                            if not HAS_OCR:
                                st.sidebar.error("Install: `pip install pillow unstructured[pillow]`")
                                raise Exception("Image OCR support missing")
                            loader = UnstructuredImageLoader(tmp_path)
                        except (subprocess.CalledProcessError, FileNotFoundError):
                            st.sidebar.error("❌ Tesseract OCR not installed. Install Tesseract to process images.")
                            raise Exception("Tesseract not available for image processing")
                    else:
                        raise ValueError("Unsupported file type")

                    docs = loader.load()
                    
                    # Update status: File loaded, now splitting
                    st.session_state.processing_status[-1]["completed"] = True
                    st.session_state.processing_status.append({
                        "message": "Splitting text into chunks for better understanding",
                        "completed": False
                    })

                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                    texts = text_splitter.split_documents(docs)

                    # Update status: Text split, now creating embeddings
                    st.session_state.processing_status[-1]["completed"] = True
                    st.session_state.processing_status.append({
                        "message": "Creating vector embeddings and storing in knowledge base",
                        "completed": False
                    })

                    # Create vector store
                    embeddings = GoogleGenerativeAIEmbeddings(
                        model="models/embedding-001",
                        google_api_key=GOOGLE_API_KEY
                    )
                    st.session_state.vector_store = FAISS.from_documents(texts, embeddings)

                    # Update status: Complete
                    st.session_state.processing_status[-1]["completed"] = True
                    st.session_state.processing_status.append({
                        "message": f"Successfully processed {uploaded_file.name} - Ready for questions!",
                        "completed": True
                    })

                    st.sidebar.success(f"✅ {uploaded_file.name} loaded!")

                except Exception as e:
                    st.sidebar.error(f"❌ Failed to process file: {str(e)}")
                    st.session_state.processing_status.append({
                        "message": f"Failed to process file: {str(e)}",
                        "completed": True
                    })
                finally:
                    os.unlink(tmp_path)

# -----------------------------
# 6. Chat Interface
# -----------------------------
# Display chat messages
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user" and msg.get("audio"):
            # User messages with audio - NO PLAYER, just text indicator
            st.write(f"🗣️ You: {msg['content']}")
        elif msg["role"] == "ai" and msg.get("tts"):
            # AI messages - KEEP PLAYER
            st.write(f"🤖 Neo: {msg['content']}")
            if not st.session_state.audio_muted:
                audio_key = f"played_{idx}"
                if audio_key not in st.session_state.played_audio_keys:
                    st.audio(msg["tts"], format="audio/mp3", autoplay=True)
                    st.session_state.played_audio_keys.add(audio_key)  # Mark as played
                else:
                    st.audio(msg["tts"], format="audio/mp3", autoplay=False)
        else:
            if msg["role"] == "user":
                st.write(f"You: {msg['content']}")
            else:
                st.write(f"🤖 Neo: {msg['content']}")

# -----------------------------
# 7. Input: Text or Voice
# -----------------------------
input_col1, input_col2 = st.columns([4, 1])

with input_col1:
    user_input = st.chat_input("Type your question to Neo...")

with input_col2:
    try:
        from streamlit_mic_recorder import mic_recorder
        # Unique key to force new recorder after processing
        audio = mic_recorder(
            start_prompt="🎤",
            stop_prompt="🛑",
            key=f"mic_{len(st.session_state.messages)}"  # Changes after new message
        )
    except:
        audio = None

# -----------------------------
# 8. Process User Input
# -----------------------------
# Initialize processed flag
if 'processed_audio_hash' not in st.session_state:
    st.session_state.processed_audio_hash = None

# Helper: hash audio bytes
def hash_audio(audio_bytes):
    import hashlib
    return hashlib.md5(audio_bytes).hexdigest()

# Flag: should we process?
should_process = False
new_transcript = None

# Case 1: Voice input
if audio and audio['bytes']:
    audio_hash = hash_audio(audio['bytes'])
    if st.session_state.processed_audio_hash != audio_hash:
        st.session_state.processed_audio_hash = audio_hash
        should_process = True
        st.session_state.processing_status = []
        
        st.session_state.processing_status.append({
            "message": "Transcribing your voice...",
            "completed": False
        })
        
        with st.spinner("Transcribing..."):
            try:
                url = "https://api.deepgram.com/v1/listen?language=en-US&punctuate=true"
                headers = {
                    "Authorization": f"Token {DEEPGRAM_API_KEY}",
                    "Content-Type": "audio/webm;codecs=opus"
                }
                response = requests.post(url, headers=headers, data=audio['bytes'], timeout=30)
                if response.status_code == 200:
                    new_transcript = response.json()['results']['channels'][0]['alternatives'][0]['transcript']
                    st.session_state.processing_status[-1]["completed"] = True
                    st.session_state.processing_status.append({
                        "message": f"Transcribed: '{new_transcript}'",
                        "completed": True
                    })
                else:
                    st.error("🎤 Transcription failed")
                    st.session_state.processing_status[-1]["completed"] = True
                    st.session_state.processing_status.append({
                        "message": "Transcription failed",
                        "completed": True
                    })
            except Exception as e:
                st.error(f"🎤 Error: {e}")
                st.session_state.processing_status[-1]["completed"] = True

# Case 2: Text input
elif user_input:
    should_process = True
    new_transcript = user_input
    st.session_state.processing_status = []

# -----------------------------
# 9. Run AI Only Once Per Input
# -----------------------------
if should_process and new_transcript:
    # Add user message
    user_msg = {"role": "user", "content": new_transcript}
    if audio and audio['bytes']:
        user_msg["audio"] = audio['bytes']
    st.session_state.messages.append(user_msg)

    # Build RAG Chain
    with st.spinner("🧠 Neo is thinking..."):
        try:
            context = ""
            if st.session_state.vector_store:
                st.session_state.processing_status.append({
                    "message": "Searching knowledge base for relevant information",
                    "completed": False
                })
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                relevant_docs = retriever.invoke(new_transcript)
                context = "\n\n".join([d.page_content for d in relevant_docs])
                st.session_state.processing_status[-1]["completed"] = True
                st.session_state.processing_status.append({
                    "message": f"Retrieved {len(relevant_docs)} relevant document sections",
                    "completed": True
                })
            else:
                st.session_state.processing_status.append({
                    "message": "No uploaded files - using general knowledge",
                    "completed": True
                })

            st.session_state.processing_status.append({
                "message": "Neo is generating response...",
                "completed": False
            })

            # Prompt with follow-up question
            prompt = ChatPromptTemplate.from_messages([
                ("system", """
                You are a helpful, kind AI Teacher.
                Your name is Neo, don't introduce yourself every time or as a AI model.
                Answer clearly and simply.
                If context is provided, use it.
                If not, say 'I don't have context for that, but here's what I know.'
                Always be polite and encouraging in your teaching.
                Explain every topic like you are explaining to a 5 year old.
                After generating the answer,summurise the answer in 2-3 lines in hinglish.
                 
                After your answer, ask a short follow-up question like:
                'What did you understand from this?' or 'Can you summarize this in your own words?' 
                to check the student's understanding.
                Keep the conversation interactive and engaging.
                
                """),
                MessagesPlaceholder(variable_name="history"),
                ("human", "Context (if relevant):\n{context}\n\nQuestion: {input}")
            ])

            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=GOOGLE_API_KEY,
                temperature=0.3,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                },
                max_output_tokens=400,
            )

            chain = (
                {
                    "context": lambda _: context,
                    "input": lambda x: x["input"],
                    "history": lambda x: x.get("history", [])
                }
                | prompt
                | llm
                | StrOutputParser()
            )

            # Build history
            history = [
                HumanMessage(content=m["content"]) if m["role"] == "user"
                else SystemMessage(content="AI Assistant") if m["role"] == "system"
                else {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]

            # Get response
            response = chain.invoke({
                "input": new_transcript,
                "history": history
            })

            st.session_state.processing_status[-1]["completed"] = True
            st.session_state.processing_status.append({
                "message": "Converting response to speech...",
                "completed": False
            })

            # TTS
            tts_bytes = None
            try:
                tts = gTTS(text=response, lang='en', slow=False)
                audio_buffer = BytesIO()
                tts.write_to_fp(audio_buffer)
                audio_buffer.seek(0)
                tts_bytes = audio_buffer.read()
                st.session_state.processing_status[-1]["completed"] = True
                st.session_state.processing_status.append({
                    "message": "Audio generated successfully!",
                    "completed": True
                })
            except Exception as e:
                st.session_state.processing_status[-1]["completed"] = True
                st.session_state.processing_status.append({
                    "message": f"Audio generation failed: {str(e)}",
                    "completed": True
                })

            # Save AI response
            ai_msg = {"role": "ai", "content": response}
            if tts_bytes:
                ai_msg["tts"] = tts_bytes
            st.session_state.messages.append(ai_msg)

            # Clear status
            time.sleep(1)
            st.session_state.processing_status = []
            st.rerun()

        except Exception as e:
            st.error(f"❌ Neo encountered an error: {str(e)}")
            st.session_state.processing_status.append({
                "message": f"Error occurred: {str(e)}",
                "completed": True
            })

# -----------------------------
# 10. Sidebar: Controls Dropdown
# -----------------------------
with st.sidebar:
    
    
    # Controls dropdown
    with st.expander("⚙️ Controls", expanded=True):
        
        st.subheader("#Insturction: Generated files will pop up at the end of the sidebar")

        # Audio controls
        st.subheader("🔊 Audio Settings")
        mute_col1, mute_col2 = st.columns(2)
        with mute_col1:
            if st.button("🔇 Mute Audio", disabled=st.session_state.audio_muted, key="sidebar_mute"):
                st.session_state.audio_muted = True
                st.rerun()
        with mute_col2:
            if st.button("🔊 Unmute Audio", disabled=not st.session_state.audio_muted, key="sidebar_unmute"):
                st.session_state.audio_muted = False
                st.rerun()
        audio_status = "🔇 Muted" if st.session_state.audio_muted else "🔊 Audio On"
        st.caption(f"Current status: {audio_status}")


        # Study Tools
        st.subheader("📘 Study Tools")

        def get_relevant_context():
            content = "\n\n".join([
                f"User: {m['content']}\nAI: {m['content']}"
                for m in st.session_state.messages
                if m["role"] == "user" or (m["role"] == "ai" and "content" in m)
            ])
            return content.strip()
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3,
            max_output_tokens=1024,               
        )

        # Generate Notes
        if st.button("📝 Generate Notes", key="gen_notes"):
            context = get_relevant_context()
            if not context:
                st.warning("No conversation to generate notes from.")
            else:
                with st.spinner("🧠 Neo is creating structured notes..."):
                    try:
                        prompt = ChatPromptTemplate.from_messages([
                            ("system", """
                            You are an expert teacher. 
                            Create clear, well-structured educational notes based on the conversation.
                            Use bullet points, headings, and simple language.
                            Focus on key concepts, definitions, and explanations.
                            Do not include greetings or meta-comments.
                            """),
                            ("human", "Conversation:\n{context}")
                        ])
                        chain = prompt | llm | StrOutputParser()
                        notes = chain.invoke({"context": context})
                        st.session_state.notes_content = notes
                        st.success("✅ Notes generated!")
                    except Exception as e:
                        st.error(f"❌ Failed: {str(e)}")

        # Generate Questions
        if st.button("❓ Generate Questions", key="gen_questions"):
            context = get_relevant_context()
            if not context:
                st.warning("No conversation to generate questions from.")
            else:
                with st.spinner("🧠 Neo is creating practice questions..."):
                    try:
                        prompt = ChatPromptTemplate.from_messages([
                            ("system", """
                            You are an expert teacher.
                            Generate 5-7 thoughtful questions based on the conversation.
                            Include a mix of:
                            - Conceptual questions
                            - Short answer
                            - Application-based
                            - One 'think deeper' question
                            Do not include answers.
                            Format: numbered list.
                            """),
                            ("human", "Conversation:\n{context}")
                        ])
                        
                        chain = prompt | llm | StrOutputParser()
                        questions = chain.invoke({"context": context})
                        st.session_state.questions_content = questions
                        st.success("✅ Questions generated!")
                    except Exception as e:
                        st.error(f"❌ Failed: {str(e)}")



        # Export Chat
        st.subheader("💾 Export Options")
        if st.session_state.messages:
            def prepare_chat_export():
                return {
                    "application": "NeoLearn",
                    "teacher": "Neo",
                    "export_date": datetime.now().isoformat(),
                    "total_messages": len(st.session_state.messages),
                    "conversation": [
                        {
                            "message_id": i + 1,
                            "role": "Student" if msg["role"] == "user" else "Neo (AI Teacher)",
                            "content": msg["content"],
                            "timestamp": datetime.now().isoformat(),
                            "has_audio": bool(msg.get("audio") or msg.get("tts"))
                        }
                        for i, msg in enumerate(st.session_state.messages)
                    ]
                }

            chat_data = prepare_chat_export()
            json_str = json.dumps(chat_data, indent=2, ensure_ascii=False)
            text_str = f"""NeoLearn Chat Export
Teacher: Neo (AI Assistant)
Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Messages: {len(st.session_state.messages)}

{'='*50}
CONVERSATION
{'='*50}

"""
            for i, msg in enumerate(st.session_state.messages):
                role = "Student" if msg["role"] == "user" else "Neo"
                audio_note = " 🎵" if msg.get("audio") or msg.get("tts") else ""
                text_str += f"[{i+1}] {role}{audio_note}: {msg['content']}\n\n"
            text_str += f"\n{'='*50}\nEnd of Chat Export\n"

            col1, col2 = st.columns(2)
            with col1:
                st.download_button("📄 JSON", json_str, f"neolearn_chat_{int(time.time())}.json", "application/json", key="export_json")
            with col2:
                st.download_button("📝 TXT", text_str, f"neolearn_chat_{int(time.time())}.txt", "text/plain", key="export_txt")
            st.caption(f"💬 {len(st.session_state.messages)} messages ready to export")





        # Personal Notes Section
        st.subheader("📝 Personal Notes")
        if st.button("📝 Open Notes", key="open_notes"):
            st.session_state.show_notes = not st.session_state.get('show_notes', False)

        if st.session_state.get('show_notes', False):
            user_notes = st.text_area(
                "Write your personal notes here:",
                value=st.session_state.get('user_notes', ''),
                height=150,
                key="user_notes_input",
                help="Take notes during your learning session"
            )
            if user_notes != st.session_state.get('user_notes', ''):
                st.session_state.user_notes = user_notes
                
            if st.button("💾 Save Notes", key="save_notes"):
                st.success("✅ Notes saved!")



        # Learning Report Section
        st.subheader("📊 Learning Report")
        if st.button("📈 Generate Learning Report", key="gen_report"):
            if not st.session_state.messages:
                st.warning("No conversation to analyze.")
            else:
                with st.spinner("🧠 Analyzing your learning patterns..."):
                    try:
                        # Prepare conversation analysis
                        conversation_analysis = []
                        question_depth_count = 0
                        follow_up_questions = 0
                        topics_covered = []
                        
                        for i, msg in enumerate(st.session_state.messages):
                            if msg["role"] == "user":
                                conversation_analysis.append(f"Student: {msg['content']}")
                                # Count follow-up questions (simple heuristic)
                                if any(word in msg['content'].lower() for word in ['why', 'how', 'what if', 'explain', 'more', 'detail']):
                                    question_depth_count += 1
                                if i > 0 and any(word in msg['content'].lower() for word in ['also', 'and', 'but', 'however', 'what about']):
                                    follow_up_questions += 1
                            elif msg["role"] == "ai":
                                conversation_analysis.append(f"Neo: {msg['content']}")

                        context = "\n".join(conversation_analysis)
                        
                        report_prompt = ChatPromptTemplate.from_messages([
                            ("system", f"""
                            You are an expert learning analyst and educational psychologist.
                            
                            Analyze this learning conversation and provide a comprehensive learning report.
                            
                            Conversation Statistics:
                            - Total messages: {len(st.session_state.messages)}
                            - Deep questions asked: {question_depth_count}
                            - Follow-up questions: {follow_up_questions}
                            - Student notes taken: {'Yes' if st.session_state.get('user_notes', '').strip() else 'No'}
                            
                            Create a detailed report with these sections:
                            
                            ## 📊 Learning Style Analysis
                            Determine if the student is:
                            - **Deep Learner** (asks many why/how questions, explores concepts thoroughly, slower but thorough)
                            - **Surface Learner** (covers topics quickly, fewer deep questions, faster but less detailed)
                            - **Balanced Learner** (mix of both approaches)
                            
                            ## 🎯 Learning Strengths
                            What they did well in this session
                            
                            ## 🔍 Areas for Improvement  
                            What they could explore deeper or ask more about
                            
                            ## ❓ Questions They Should Have Asked
                            List 3-5 important questions about the topics discussed that would deepen understanding
                            
                            ## 📚 Study Recommendations
                            Personalized suggestions based on their learning style:
                            - Study techniques that match their style
                            - How to improve their weaker areas
                            - Specific next steps for continued learning
                            
                            ## 🏆 Overall Assessment
                            Summary and encouragement
                            
                            Be encouraging, specific, and actionable. Use their actual conversation content as evidence.
                            """),
                            ("human", "Learning Session Conversation:\n{context}")
                        ])
                        
                        chain = report_prompt | llm | StrOutputParser()
                        report = chain.invoke({"context": context})
                        st.session_state.learning_report = report
                        st.success("✅ Learning report generated!")
                        
                    except Exception as e:
                        st.error(f"❌ Failed to generate report: {str(e)}")



        # Clear button
        st.subheader("🧹 Reset")
        if st.button("🧹 Clear Chat & Files", key="clear_all"):
            keys_to_clear = ['messages', 'vector_store', 'last_file_key', 'processing_status',
                             'notes_content', 'questions_content', 'played_audio_keys', 'processed_audio_hash',
                             'user_notes', 'learning_report', 'show_notes']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    # Display generated content outside the expander
    if hasattr(st.session_state, 'notes_content'):
        with st.expander("📘 Generated Notes", expanded=False):
            st.write(st.session_state.notes_content)
            st.download_button("📥 Download Notes", st.session_state.notes_content, "neolearn_notes.txt", "text/plain", key="download_notes")

    if hasattr(st.session_state, 'questions_content'):
        with st.expander("❓ Generated Questions", expanded=False):
            st.write(st.session_state.questions_content)
            st.download_button("📥 Download Questions", st.session_state.questions_content, "neolearn_questions.txt", "text/plain", key="download_questions")

    # Display Personal Notes
    if st.session_state.get('user_notes', '').strip():
        with st.expander("📝 Your Personal Notes", expanded=False):
            st.write(st.session_state.user_notes)
            st.download_button("📥 Download Personal Notes", st.session_state.user_notes, "my_personal_notes.txt", "text/plain", key="download_personal_notes")

    # Display Learning Report
    if hasattr(st.session_state, 'learning_report'):
        with st.expander("📊 Learning Report", expanded=True):
            st.markdown(st.session_state.learning_report)
            
            # Prepare detailed report for download
            report_with_metadata = f"""NeoLearn - Personal Learning Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Student Session Analysis

{'='*60}

{st.session_state.learning_report}

{'='*60}

Session Statistics:
- Total Messages: {len(st.session_state.messages)}
- Personal Notes Taken: {'Yes' if st.session_state.get('user_notes', '').strip() else 'No'}
- Knowledge Base Used: {'Yes' if st.session_state.vector_store else 'No'}

Generated by NeoLearn AI Learning Assistant
"""
            st.download_button("📥 Download Learning Report", report_with_metadata, f"learning_report_{int(time.time())}.txt", "text/plain", key="download_report")

    # Knowledge base info

    if st.session_state.vector_store:
        st.success("📚 Knowledge base loaded")
        st.caption("Neo can answer questions about your uploaded file")
    else:
        st.info("📝 No files uploaded")
        st.caption("Upload a file to expand Neo's knowledge")


    
    st.caption(f"💬 Messages: {len(st.session_state.messages)}")
    st.caption("Developed with ❤️ by Dhruv Bhardwaj")
# Add CSS
st.markdown("""
<style>
.stAudio > div { margin: 0.5rem 0; }
.processing-status { background-color: #f0f2f6; padding: 0.5rem; border-radius: 0.5rem; margin: 0.25rem 0; }
</style>
""", unsafe_allow_html=True)