# LUAWMS AI Assistant

An AI-powered chatbot designed for **Lasbela University of Agriculture, Water and Marine Sciences (LUAWMS)**. This assistant leverages **Retrieval-Augmented Generation (RAG)** to provide accurate information regarding university admissions, fees, departments, and secure student records to students, visitors, and administration.

## 🚀 Features

- **Intelligent RAG Pipeline**: Queries a dedicated knowledge base (`knowledge_base.json`) to answer general university questions with high accuracy.
- **Secure Student Portal**: Authenticated access for students to view academic results, CGPA, and semester details.
- **Dynamic Leaderboards**: Automatically generates class standings and merit lists based on batch data (e.g., "2k21").
- **Universal Search**: Administrative capability to search for specific student records.
- **Context-Aware**: Maintains session context for a natural conversational experience.

## 🛠️ Tech Stack

- **Framework**: FastAPI
- **LLM Engine**: Groq (Llama 3.1)
- **Embeddings**: Sentence Transformers (`all-MiniLM-L6-v2`)
- **Database**: SQLite (Session management)
- **Vector Search**: Scikit-learn (Cosine Similarity)

## 📦 Setup & Installation

1.  **Clone the repository**
    ```bash
    git clone <repository-url>
    cd luawms-ai
    ```

2.  **Create a virtual environment**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment**
    Create a `.env` file in the root directory and add your Groq API key:
    ```env
    GROQ_API_KEY=your_groq_api_key_here
    ```

5.  **Prepare Data**
    - `knowledge_base.json`: Pre-loaded with public university information.
    - `students.json`: Required for student features. Rename `students.sample.json` to `students.json` and populate it with actual student records.

## ⚡ Usage

### Start the Server
Run the FastAPI application using the provided entry point:

```bash
python main.py
```
The API will be available at `http://127.0.0.1:8000`.

## Project Structure
- `main.py`: Core application logic and API endpoints.
- `knowledge_base.json`: Source data for RAG.
