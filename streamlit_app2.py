import streamlit as st
from transformers import pipeline
from streamlit_mic_recorder import mic_recorder, speech_to_text
import torch  # Ensure torch is imported
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Load embeddings and vector database
embeddings_10_percent = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'token': st.secrets["API_KEY1"]}
)

vector_db_10_percent = Chroma(
    persist_directory="/content/drive/MyDrive/chroma_db_10_percent_drive", 
    embedding_function=embeddings_10_percent,
    collection_name="subtitle_embeddings_10_percent" 
)

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-002",
    temperature=0.3,
    google_api_key=st.secrets["API_KEY2"]
)

# Create retriever and prompt template
retriever = vector_db_10_percent.as_retriever(search_type="similarity", search_kwargs={"k": 5})
prompt_template = """Use the following movie subtitle excerpts to answer the question.
Focus on dialogues and emotional context. If unsure, state "Insufficient context".

Context:
{context}

Question: {question}
Answer:"""
rag_prompt = ChatPromptTemplate.from_template(prompt_template)

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": rag_prompt},
    return_source_documents=True
)

st.title("Movie Subtitle Q&A (Voice + Text)")

question = ""  

input_mode = st.radio("Choose your input method:", ["Text", "Audio"])

if input_mode == "Text":
    question = st.text_input("Enter your question:", key="text_input")
    
elif input_mode == "Audio":
    st.write("Click below to record your question:")
    
    audio = mic_recorder(
        start_prompt="Start recording",
        stop_prompt="Stop recording",
        key="recorder"
    )
    
    if audio and audio.get("bytes"):
        # Play recorded audio for confirmation
        st.audio(audio["bytes"], format="audio/wav")
        
        try:
            transcriber = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-base",
                device="cuda:0" if torch.cuda.is_available() else "cpu"
            )
            transcription_result = transcriber(audio["bytes"])
            question = transcription_result["text"]
            
            question = st.text_input("Transcribed Question:", value=question, key="audio_transcription")
        
        except Exception as e:
            st.error(f"Transcription failed: {str(e)}")

if st.button("Submit") and question:
    result = rag_chain.invoke({"query": question})
    
    st.write(f"**Answer:** {result['result']}")

    # Display relevant contexts if available
    if result['source_documents']:
        st.write("**Relevant Contexts:**")
        for doc in result['source_documents']:
            st.write(f"- {doc.page_content[:150]}...")
            if doc.metadata:
                st.write(f"  Metadata: {doc.metadata}")
    else:
        st.write("No relevant contexts found. Try rephrasing your question or expanding the dataset.")
