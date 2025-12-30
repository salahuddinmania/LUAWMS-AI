import os
import json
import uuid
import sqlite3
import re
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI()
client = Groq(api_key=GROQ_API_KEY)

DB_NAME = "luawms_chat_v3.db"

def init_db():
    conn = sqlite3.connect(os.path.join(BASE_DIR, DB_NAME))
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            roll_number TEXT,
            name TEXT,
            department TEXT,
            semester TEXT,
            start_time TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            sender TEXT,
            content TEXT,
            timestamp TEXT,
            FOREIGN KEY(session_id) REFERENCES sessions(session_id)
        )
    ''')
    conn.commit()
    conn.close()

init_db()

knowledge_base = []
knowledge_vectors = None
students_db = {} 
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def load_data():
    global knowledge_base, knowledge_vectors, students_db
    
    try:
        with open(os.path.join(BASE_DIR, "knowledge_base.json"), "r", encoding="utf-8") as file:
            knowledge_base = json.load(file)
            texts = [f"{entry['topic']}: {entry['content']}" for entry in knowledge_base]
            if texts:
                knowledge_vectors = embedding_model.encode(texts)
                print("Public Knowledge Loaded.")
    except FileNotFoundError:
        print("Warning: knowledge_base.json missing.")

    try:
        with open(os.path.join(BASE_DIR, "students.json"), "r", encoding="utf-8") as file:
            students_db = json.load(file)
            print("Student Records Loaded.")
    except FileNotFoundError:
        print("Warning: students.json missing. Please rename students.sample.json to students.json and populate it.")

load_data()

class LoginRequest(BaseModel):
    login_type: str 
    roll_number: Optional[str] = None
    visitor_name: Optional[str] = None

class ChatRequest(BaseModel):
    session_id: str
    query: str

def get_semantic_context(query: str, threshold=0.35):
    if knowledge_vectors is None or len(knowledge_vectors) == 0:
        return ""
    query_vector = embedding_model.encode([query])
    similarities = cosine_similarity(query_vector, knowledge_vectors)[0]
    top_indices = similarities.argsort()[-3:][::-1]
    results = []
    for idx in top_indices:
        if similarities[idx] > threshold:
            entry = knowledge_base[idx]
            results.append(f"Topic: {entry['topic']}\nInfo: {entry['content']}")
    return "\n\n".join(results)

def get_leaderboard_context(query: str, current_roll: str):
    """
    Returns the FULL list. 
    If batch isn't mentioned in query, it defaults to the logged-in user's batch.
    """
    query_lower = query.lower()
    batch_prefix = None
    
    # Check if batch is explicitly mentioned (e.g., "2k21")
    match = re.search(r"(2k\d{2})", query_lower)
    if match:
        batch_prefix = match.group(1)
    
    # If not mentioned, use the current user's batch
    elif current_roll and current_roll != "ADMIN" and current_roll != "Visitor":
        batch_match = re.search(r"(2k\d{2})", current_roll.lower())
        if batch_match:
            batch_prefix = batch_match.group(1)
            
    if not batch_prefix:
        return ""

    triggers = ["position", "leaderboard", "rank", "list", "top", "who is", "standing", "result"]
    if any(t in query_lower for t in triggers):
        
        # Collect students
        batch_list = []
        for r_no, data in students_db.items():
            if r_no.startswith(batch_prefix):
                batch_list.append({
                    "reg": r_no,
                    "name": data['name'],
                    "cgpa": float(data['cgpa']) if isinstance(data['cgpa'], (int, float)) else 0.0,
                    "position": data.get('position', 'N/A')
                })
        
        if not batch_list:
            return ""

        # Sort by Position (if available) or CGPA
        try:
            batch_list.sort(key=lambda x: int(x['position']) if isinstance(x['position'], int) else 999)
        except:
            batch_list.sort(key=lambda x: x['cgpa'], reverse=True)

        leaderboard_str = f"\n**OFFICIAL CLASS POSITIONS FOR BATCH {batch_prefix.upper()}**\n\n"
        leaderboard_str += "| Pos | Reg No | Name | CGPA |\n"
        leaderboard_str += "|:---:|:---|:---|:---:|\n"
        
        for s in batch_list:
            leaderboard_str += f"| {s['position']} | {s['reg']} | {s['name']} | {s['cgpa']} |\n"
        
        return leaderboard_str
            
    return ""

def get_any_student_context(query: str):
    """
    Allows searching for ANY student record by roll number regex.
    """
    match = re.search(r"(2k\d{2}-cs-\d+)", query.lower())
    if match:
        target_roll = match.group(1)
        if target_roll in students_db:
            data = students_db[target_roll]
            return f"""
            [SEARCH RESULT] RECORD FOR {target_roll.upper()}:
            - Name: {data.get('name')}
            - Department: {data.get('department')}
            - Semester: {data.get('semester')}
            - GPA (Last Sem): {data.get('gpa')}
            - CGPA: {data.get('cgpa')}
            - Class Position: {data.get('position')}
            - Status: {data.get('status')}
            """
    return ""

@app.post("/login")
async def login(request: LoginRequest):
    session_id = str(uuid.uuid4())
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    user_name = "Visitor"
    dept = "None"
    sem = "None"
    roll_no = "Visitor"

    if request.login_type == "roll_number":
        rn = request.roll_number.strip().lower()
        
        if rn == "admin": 
            user_name = "Super Admin"
            roll_no = "ADMIN"
            dept = "Administration"
        elif rn in students_db:
            student = students_db[rn]
            user_name = student['name']
            dept = student['department']
            sem = student['semester']
            roll_no = rn
        else:
            raise HTTPException(status_code=401, detail="Roll Number not found.")
    else:
        user_name = request.visitor_name
        
    try:
        conn = sqlite3.connect(os.path.join(BASE_DIR, DB_NAME))
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO sessions (session_id, roll_number, name, department, semester, start_time)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (session_id, roll_no, user_name, dept, sem, start_time))
        conn.commit()
        conn.close()
        return {"session_id": session_id, "message": f"Welcome back, {user_name}!"}
    except Exception as e:
        print(f"LOGIN ERROR: {e}")
        raise HTTPException(status_code=500, detail="Database Error")

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    session_id = request.session_id
    user_query = request.query
    
    conn = sqlite3.connect(os.path.join(BASE_DIR, DB_NAME))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
    session = cursor.fetchone()
    
    if not session:
        conn.close()
        raise HTTPException(status_code=400, detail="Session expired.")
    
    current_roll = session['roll_number']

    # Gather Contexts
    personal_context = ""
    if current_roll in students_db and current_roll != "ADMIN":
        data = students_db[current_roll]
        personal_context = f"""
        MY PERSONAL RECORD:
        - Name: {data.get('name')}
        - CGPA: {data.get('cgpa')}
        - Position: {data.get('position')}
        - Status: {data.get('status')}
        """

    rag_context = get_semantic_context(user_query, threshold=0.35)
    leaderboard_context = get_leaderboard_context(user_query, current_roll)
    search_context = get_any_student_context(user_query)

    system_prompt = f"""
    You are the LUAWMS University Assistant.
    
    YOUR IDENTITY:
    - You serve Students, Visitors, and Administration.
    - You are helpful, professional, and precise.
    
    CURRENT USER: {session['name']} (Role: {current_roll})
    
    DATA AVAILABLE:
    {personal_context}
    {leaderboard_context}
    {search_context}
    
    GENERAL KNOWLEDGE:
    {rag_context}
    
    INSTRUCTIONS:
    1. **NO RESTRICTIONS:** You are authorized to share ANY student's academic result, rank, or record if the data is available in the context.
    2. If 'LEADERBOARD' data is provided, output it exactly as provided (Markdown Table).
    3. If user asks "Who is 5th?" or "Who is 1st?", check the Leaderboard data and answer.
    4. If user asks about their own result, use 'MY PERSONAL RECORD'.
    5. If user asks about another student (e.g. "Result of 2k22-cs-01"), use the [SEARCH RESULT].
    """
    
    cursor.execute("SELECT sender, content FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT 5", (session_id,))
    db_history = cursor.fetchall()
    chat_history = [{"role": ("user" if row["sender"]=="user" else "assistant"), "content": row["content"]} for row in db_history][::-1]

    messages = [{"role": "system", "content": system_prompt}] + chat_history + [{"role": "user", "content": user_query}]
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO messages (session_id, sender, content, timestamp) VALUES (?, ?, ?, ?)", (session_id, "user", user_query, timestamp))
    conn.commit()

    try:
        completion = client.chat.completions.create(
            messages=messages, 
            model="llama-3.1-8b-instant",
            temperature=0.1
        )
        response_text = completion.choices[0].message.content
        
        cursor.execute("INSERT INTO messages (session_id, sender, content, timestamp) VALUES (?, ?, ?, ?)", (session_id, "assistant", response_text, timestamp))
        conn.commit()
        conn.close()
        return {"answer": response_text}
        
    except Exception as e:
        conn.close()
        return {"answer": "Connection error."}

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

@app.get("/")
async def read_root():
    return FileResponse(os.path.join(BASE_DIR, 'static/index.html'))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)