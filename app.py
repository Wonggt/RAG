import streamlit as st
import os
import dotenv
import uuid
import re

# check if it's linux so it works on Streamlit Cloud
if os.name == 'posix':
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, AIMessage

from rag_methods import (
    load_doc_to_db,
    load_url_to_db,
    stream_llm_response,
    stream_llm_rag_response,
    get_retriever,
    get_optimized_retriever,
)
from agentic_rag import build_agentic_rag_graph, run_agentic_rag

dotenv.load_dotenv()

MODELS = [
        "inclusionai/ling-3.0-tiny:free",
        "nvidia/llama-nemotron-rerank-vl-1b-v2:free",
        "google/gemma-4-26b-a4b-it:free",
        "google/gemma-4-31b-it:free",
]

st.set_page_config(
    page_title="RAG LLM app?", 
    page_icon="📚", 
    layout="centered", 
    initial_sidebar_state="expanded"
)


from tts import generate_tts





# --- Header ---
st.html("""<h2 style="text-align: center;">📚🔍 <i> Please ask me something... </i> 🤖💬</h2>""")


# --- Initial Setup ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there! How can I assist you today?"}
]
# 在初始设置部分添加一个新的session_state来存储TTS音频文件
if "tts_audio_files" not in st.session_state:
    st.session_state.tts_audio_files = []


# ✅ 新增這一行
if "model" not in st.session_state:
    st.session_state.model = MODELS[0]
    
# --- Side Bar LLM API Tokens ---
with st.sidebar:
    if "AZ_OPENAI_API_KEY" not in os.environ:
        default_openrouter_api_key = os.getenv("OPENROUTER_API_KEY") if os.getenv("OPENROUTER_API_KEY") is not None else ""  # only for development environment, otherwise it should return None
        with st.popover("🔐 OpenRouter"):
            openrouter_api_key = st.text_input(
                "Introduce your OpenRouter API Key (https://openrouter.ai/)", 
                value=default_openrouter_api_key, 
                type="password",
                key="openrouter_api_key",
            )

    else:
        openrouter_api_key = None
        st.session_state.openrouter_api_key = None



# --- Main Content ---
# Checking if the user has introduced the OpenAI API Key, if not, a warning is displayed
missing_openai = openrouter_api_key == "" or openrouter_api_key is None or "sk-" not in openrouter_api_key
if missing_openai :
    st.write("#")
    st.warning("⬅️ Please introduce an API Key to continue...")

else:
    # Sidebar
    with st.sidebar:
        st.divider()
        models = []
        for model in MODELS:
            if not missing_openai:
                models.append(model)

        st.selectbox(
            "🤖 Select a Model", 
            options=models,
            key="model",
        )

        cols0 = st.columns(2)
        with cols0[0]:
            is_vector_db_loaded = ("vector_db" in st.session_state and st.session_state.vector_db is not None)
            st.toggle(
                "Use RAG",
                value=is_vector_db_loaded,
                key="use_rag",
                disabled=not is_vector_db_loaded,
            )
            st.toggle(
                "🧠 Agentic mode",
                value=False,
                key="use_agentic",
                disabled=not is_vector_db_loaded,
                help="Query rewriting + relevance grading + citations (slower, higher quality)",
            )

        with cols0[1]:
            def clear_chat():
                if "tts_audio_files" not in st.session_state:
                    st.session_state.tts_audio_files = []

                st.session_state.messages = [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there! How can I assist you today?"}
                ]
                st.session_state.tts_audio_files.clear()
            
            st.button("Clear Chat", on_click=clear_chat, type="primary")

        st.header("RAG Sources:")
            
        # File upload input for RAG with documents
        st.file_uploader(
            "📄 Upload a document", 
            type=["pdf"],
            accept_multiple_files=True,
            on_change=load_doc_to_db,
            key="rag_docs",
        )

        # URL input for RAG with websites
        st.text_input(
            "🌐 Introduce a URL", 
            placeholder="https://example.com",
            on_change=load_url_to_db,
            key="rag_url",
        )

        with st.expander(f"📚 Documents in DB ({0 if not is_vector_db_loaded else len(st.session_state.rag_sources)})"):
            st.write([] if not is_vector_db_loaded else [source for source in st.session_state.rag_sources])

    
    # Main chat app
    # model_provider = st.session_state.model.split("/")[0]

    llm_stream = ChatOpenAI(
        api_key=openrouter_api_key,
        model_name=st.session_state.model,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.2,
        streaming=True,
    )
    # elif model_provider == "anthropic":
    #     llm_stream = ChatAnthropic(
    #         api_key=openrouter_api_key,
    #         model=st.session_state.model.split("/")[-1],
    #         temperature=0.3,
    #         streaming=True,
    #     )


   #for message in st.session_state.messages:
     #   with st.chat_message(message["role"]):
      #      st.markdown(message["content"])'''
   # '''for message in st.session_state.messages:
     #   with st.chat_message(message["role"]):
       #     st.markdown(message["content"])'''
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # 如果是助手消息且有对应的音频，则显示音频
            if message["role"] == "assistant":
                # Get the message content to match with audio files
                message_preview = message["content"][:50] + "..."
                # Find the matching audio file by content instead of index
                matching_audio = next((audio for audio in st.session_state.tts_audio_files 
                                     if audio["message"] == message_preview), None)
                if matching_audio:
                    st.audio(matching_audio["audio"], format="audio/wav")

    if prompt := st.chat_input("Your message"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            messages = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in st.session_state.messages]

            # Route: Agentic RAG > classic RAG > plain LLM
            use_agentic = st.session_state.get("use_agentic", False) and st.session_state.get("use_rag", False)

            if use_agentic:
                # Agentic pipeline is not token-streamed (each node runs to completion).
                # Show a status spinner so the user sees progress instead.
                with st.status("🧠 Agentic RAG thinking...", expanded=True) as status:
                    # Optimized retrieval: BM25 + FAISS hybrid -> cross-encoder rerank
                    chunks = st.session_state.get("all_chunks", [])
                    if chunks:
                        retriever = get_optimized_retriever(
                            st.session_state.vector_db, chunks, top_n=4
                        )
                        st.write("🔀 Using Hybrid (BM25 + FAISS) + BGE Rerank")
                    else:
                        retriever = get_retriever(st.session_state.vector_db, k=4)
                        st.write("📥 Using FAISS-only retriever (no chunks tracked)")
                    graph = build_agentic_rag_graph(llm_stream, retriever)
                    result = run_agentic_rag(graph, prompt)
                    for step in result["trace"]:
                        st.write(step)
                    status.update(label="✅ Done", state="complete", expanded=False)

                full_response = result["answer"]
                message_placeholder.markdown(full_response)

                # Render citations
                if result["citations"]:
                    with st.expander(f"📎 Sources ({len(result['citations'])})"):
                        for c in result["citations"]:
                            page = f" (p.{c['page']+1})" if c["page"] is not None else ""
                            src = os.path.basename(str(c["source"]))
                            st.markdown(f"**[{c['index']}] {src}{page}**")
                            st.caption(c["snippet"] + "...")
            else:
                if not st.session_state.use_rag:
                    response_stream = stream_llm_response(llm_stream, messages)
                else:
                    response_stream = stream_llm_rag_response(llm_stream, messages)

                # Collect and display streamed response
                for chunk in response_stream:
                    full_response += chunk
                    message_placeholder.markdown(full_response)


            # Generate TTS audio and display it
            # print("full response:"+ full_response)
            #tts_audio,lang = generate_tts(full_response) ori
            #st.audio(tts_audio, format="audio/wav")ori
            cleaned_response = re.sub(r'[*#]', '', full_response)

            tts_audio, lang = generate_tts(cleaned_response)
            
            # 保存TTS音频文件的引用
            #'''if tts_audio:
               # st.session_state.tts_audio_files.append({"audio": tts_audio, "message": full_response[:50] + "..."})
            
            # 显示当前音频
            #st.audio(tts_audio, format="audio/wav")
            
            # 将当前回复添加到消息历史
            #st.session_state.messages.append({"role": "assistant", "content": full_response})'''
            if tts_audio:
                message_preview = full_response[:50] + "..."
                st.session_state.tts_audio_files.append({"audio": tts_audio, "message": message_preview})
                # 显示当前音频（gTTS 输出的是 mp3）
                st.audio(tts_audio, format="audio/mp3")
            else:
                st.info("🔇 语音生成暂不可用（可能是不支持的语言或网络问题）")
            
            # 将当前回复添加到消息历史（这行已经存在，不需要重复添加）
            st.session_state.messages.append({"role": "assistant", "content": full_response})




with st.sidebar:
    st.divider()


