# 🧠 RAG Knowledge Assistant

A **Retrieval-Augmented Generation (RAG)**-based intelligent assistant
built using **React + TypeScript** for the frontend, **FastAPI** for the
backend, and **LLaMA** from **Hugging Face** as the open-source language
model.\
It integrates **vector-based document retrieval** with **natural
language understanding** to answer user queries based on your knowledge
base.

------------------------------------------------------------------------

## 🚀 Features

-   ⚡ **Real-time Q&A** using RAG pipeline
-   🔍 **Semantic Search** powered by vector embeddings
-   🧠 **LLaMA Integration** (Hugging Face open-source model)
-   🎨 **Modern UI** built with **React + Tailwind CSS**
-   🌐 **REST APIs** using FastAPI backend
-   📄 **Multi-format document ingestion** (PDF, DOCX, TXT)

------------------------------------------------------------------------

## 🛠️ Tech Stack

  Component       Technology
  --------------- ----------------------------------
  **Frontend**    React + TypeScript + TailwindCSS
  **Backend**     FastAPI (Python)
  **Vector DB**   FAISS
  **LLM**         Hugging Face LLaMA
  **Embedding**   Hugging Face Transformers
  **Styling**     Tailwind CSS

------------------------------------------------------------------------

## 🧩 Architecture

``` mermaid
flowchart TD
    A[User Query] --> B[React + Tailwind Frontend]
    B --> C[FastAPI Backend]
    C --> D[Embedding Generator]
    D --> E[Vector DB - FAISS]
    E --> F[Relevant Docs]
    C --> G[LLaMA Model - Hugging Face]
    F --> G
    G --> H[Final Answer Returned]
    H --> A
```

------------------------------------------------------------------------

## ⚙️ Installation

### 1️⃣ Clone the repository

``` bash
git clone https://github.com/your-username/rag-knowledge-assistant.git
cd rag-knowledge-assistant
```

### 2️⃣ Install dependencies

#### Frontend

``` bash
cd frontend
npm install
```

#### Backend

``` bash
cd backend
pip install -r requirements.txt
```

### 3️⃣ Set up environment variables

Create a `.env` file in your **backend** folder:

``` env
HUGGINGFACE_API_KEY=your_huggingface_token
MODEL_NAME=meta-llama/Llama-2-7b-chat-hf
VECTOR_DB_PATH=./vector_store
```

------------------------------------------------------------------------

## ▶️ Run the Project

### **Backend**

``` bash
cd backend
uvicorn app:app --reload
```

### **Frontend**

``` bash
cd frontend
npm run dev
```

Open: **http://localhost:5173**

## Create Virtual Environment

``` bash
python -m venv venv
venv\Scripts\activate
```

------------------------------------------------------------------------

## 📂 Project Structure

    rag-knowledge-assistant/
    ├── backend/
    │   ├── main.py                # FastAPI app
    │   ├── ingestion.py           # Document ingestion pipeline
    │   ├── retrieval.py           # RAG + LLaMA integration
    │   ├── requirements.txt
    │   └── .env
    ├── frontend/
    │   ├── src/
    │   │   ├── components/        # Chat UI components
    │   │   ├── App.tsx
    │   │   └── index.css
    │   ├── package.json
    │   └── vite.config.ts
    └── README.md

------------------------------------------------------------------------

## 🔮 Future Enhancements

-   🗂 **Multi-file document ingestion**
-   🧩 **Support for OpenAI + Mixtral models**
-   📊 **Analytics dashboard**
-   🌍 **Deployment-ready Docker setup**

------------------------------------------------------------------------

## 🤝 Contributing

Pull requests are welcome! For significant changes, please open an issue
first.
