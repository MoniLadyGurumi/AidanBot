# 🧸 Lady Gurumi's AidanBot

Welcome to **Lady Gurumi’s AidanBot** — a snuggly support assistant powered by GPT-3.5 and infused with plushie wisdom, yarn-based wit, and cat fur (optional, but inevitable).
I created this repo to upload my <em>MIT GenAI Course Assignment 3.1 - "Building a Sales And Support Agent with LangChain"</em>.
Please, be kind, I am learning 💜

This Streamlit app answers all your frequently asked questions about <a href="https://www.instagram.com/lady__gurumi">Lady Gurumi</a>’s handmade amigurumi plushies — including returns, shipping, materials, and the emotional state of Batman.
<br>
## ✨ Features

- 💬 Scrollable chat interface that keeps your cozy convo in view
- 🧶 Uses LangChain to search pre-written answers in `support_qa.txt`
- 🤖 Powered by GPT-3.5 via OpenAI
- 🎀 Fully styled with pastel flair and a irresistible banner
- 📦 Local dev server OR deployable on Streamlit Cloud
    
## 📁 Project Structure

- aidanbot_app.py # Main Streamlit app
- support_qa.txt # AidanBot's knowledge base
- lady_gurumi_banner.png # Pastel banner illustration
- requirements.txt # App dependencies

## 🪡 Technologies Used

-  UI --> Streamlit
-  LLM --> OpenAI GPT-3.5-Turbo
-  RAG --> LangChain’s `RetrievalQAWithSourcesChain`
-  Embedding --> `SentenceTransformerEmbeddings` (MiniLM)
-  Vector DB --> FAISS
-  Deployment-ready --> GitHub + Streamlit Cloud

## 🚀 Running Locally

1. Clone this repo  
2. Create a virtual environment  
3. Install dependencies  
4. Add your `.env` file with this line: OPENAI_API_KEY=your-api-key-here
5. Run: streamlit run aidanbot_app.py

## 🧭 System Design Steps

To build a system capable of handling sales-related inquiries and customer support for a boutique business, I followed this structured approach:

### 1. Define Use Cases
- Answer shipping, return, and product-related questions
- Maintain brand voice and emotional tone
- Handle multiple user interactions in sequence

### 2. Prepare Contextual Knowledge
- Created a file (`support_qa.txt`) with real-world FAQs
- Enhanced answers with brand tone and humor

### 3. Set Up RAG Pipeline
- Used LangChain’s `TextLoader` to ingest the text
- Split documents into overlapping chunks with `RecursiveCharacterTextSplitter`
- Embedded chunks with `MiniLM` via `SentenceTransformerEmbeddings`
- Stored them in `Chroma` vector store

### 4. Define Prompt Persona
- Used `PromptTemplate` to shape AidanBot's tone and behavior
- Emphasized helpfulness, humor, and “only escalate to Moni if needed”

### 5. Connect LLM
- Used `ChatOpenAI` with temp=0.4 for friendly, reliable output

### 6. Build Frontend
- Created a chat UI with `streamlit`
- Styled banner, inputs, and scrollable history box
- Preserved chat memory in `st.session_state`

---

## 🧩 System Architecture Diagram
```plaintext
User (web browser)
   |
 Streamlit UI (chat input/output)
   |
┌──────────────────────────┐
│     LangChain Chain      │
├──────────────────────────┤
│ PromptTemplate (AidanBot)│
│    ↕                    ↕
│ ChatOpenAI         Vector Retriever
│                     ↳ Chroma DB
│                     ↳ MiniLM Embeddings
│                     ↳ support_qa.txt
└──────────────────────────┘
```

## 🪪 License

This project is licensed under the MIT License, allowing others to use, modify, and share the code with attribution.

## 💌 Contact

Moni (<a href="https://www.instagram.com/lady__gurumi">Lady Gurumi</a> herself 🧶)
