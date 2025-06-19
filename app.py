import streamlit as st
import sqlite3
import os
from docx import Document # Para leer DOCX
from PyPDF2 import PdfReader # Para leer PDF
from datetime import datetime
import pandas as pd # Para mostrar la tabla de documentos de forma más bonita
import google.generativeai as genai # Nueva Importación para Google Gemini
import textwrap # Para formatear la salida del texto
from tenacity import retry, stop_after_attempt, wait_fixed # Para reintentos en llamadas a la API
import numpy as np # <-- AÑADIDO: Para cálculos de similitud de vectores
import json # <-- AÑADIDO: Para serializar/deserializar embeddings a/desde BLOB

# --- Configuración de la base de datos y IA ---
DB_FILE = 'documentos_legales.db' # Nombre del archivo de tu base de datos SQLite
TARGET_DIR = "documentos_originales" # Carpeta para guardar los archivos subidos

# Asegurarse de que la carpeta de destino exista al inicio de la aplicación
os.makedirs(TARGET_DIR, exist_ok=True)

# Configurar la API de Google Gemini (usando st.secrets para mayor seguridad)
# Asegúrate de que tu archivo .streamlit/secrets.toml tiene GOOGLE_API_KEY="TU_CLAVE"
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception as e:
    st.error(f"Error al configurar la API de Google Gemini. Asegúrate de que GOOGLE_API_KEY esté en .streamlit/secrets.toml. Error: {e}")
    st.stop() # Detiene la ejecución si la API no se puede configurar

# Inicializar el modelo Gemini Pro (o el modelo que prefieras)
# Puedes elegir otros modelos como 'gemini-pro-vision' para multimodal, pero 'gemini-pro' es para texto
# Considera 'gemini-1.5-flash' o 'gemini-1.5-pro' para mejor rendimiento y costos si tu clave lo permite.
model = genai.GenerativeModel('models/gemini-pro') 

# --- Funciones de Utilidad para Documentos ---
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS documentos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filepath TEXT NOT NULL UNIQUE,
            filename TEXT NOT NULL,
            full_text TEXT,
            doc_type TEXT,
            legal_area TEXT,
            parties TEXT,
            doc_date TEXT,
            summary TEXT,
            keywords TEXT,
            last_updated TEXT
        )
    ''')
    # NUEVA TABLA: document_chunks <-- MODIFICADO (asegurándose que esta tabla exista)
    c.execute('''
        CREATE TABLE IF NOT EXISTS document_chunks (
            chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            embedding BLOB, -- BLOB para almacenar el embedding como bytes
            FOREIGN KEY (document_id) REFERENCES documentos(id)
        )
    ''')
    conn.commit()
    conn.close()

def get_docx_text(filepath):
    text = ""
    try:
        doc = Document(filepath)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        st.error(f"Error al extraer texto de DOCX {filepath}: {e}")
        return None
    return text

def get_pdf_text(filepath):
    text = ""
    try:
        with open(filepath, "rb") as file:
            reader = PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        st.error(f"Error al extraer texto de PDF {filepath}: {e}")
    return text

def insert_document(filepath, filename, full_text, doc_type="", legal_area="", parties="", doc_date="", summary="", keywords=""):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        c.execute("SELECT id FROM documentos WHERE filepath = ?", (filepath,))
        existing_doc = c.fetchone()
        if existing_doc:
            st.warning(f"El documento '{filename}' ya existe en la base de datos. Se omitirá. Considera borrarlo y cargarlo de nuevo si cambiaste su contenido.")
            return False

        full_text_to_insert = full_text if full_text is not None else ""

        # Insertar en la tabla de documentos
        c.execute('''
            INSERT INTO documentos (filepath, filename, full_text, doc_type, legal_area, parties, doc_date, summary, keywords, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (filepath, filename, full_text_to_insert, doc_type, legal_area, parties, doc_date, summary, keywords, datetime.now().isoformat()))
        
        document_id = c.lastrowid # Obtener el ID del documento recién insertado

        # Generar chunks y embeddings para el nuevo documento <-- MODIFICADO (nueva lógica de inserción de chunks)
        chunks = get_document_chunks(full_text_to_insert)
        st.info(f"Generando embeddings para {len(chunks)} trozos de '{filename}'...")
        for i, chunk_text in enumerate(chunks):
            embedding = get_text_embedding(chunk_text)
            if embedding is not None:
                # Almacenar el embedding como bytes (BLOB) para SQLite
                embedding_bytes = json.dumps(embedding).encode('utf-8')
                c.execute('''
                    INSERT INTO document_chunks (document_id, chunk_text, embedding)
                    VALUES (?, ?, ?)
                ''', (document_id, chunk_text, embedding_bytes))
            else:
                st.warning(f"No se pudo generar embedding para el trozo {i+1} de '{filename}'.")
        
        conn.commit()
        st.success(f"Documento '{filename}' procesado e indexado con {len(chunks)} trozos.")
        return True
    except sqlite3.IntegrityError:
        st.warning(f"Error de integridad con '{filename}'. Puede que ya exista.")
        return False
    except Exception as e:
        st.error(f"Error al añadir documento '{filename}': {e}")
        return False
    finally:
        conn.close()

def search_documents(query, limit=20):
    # NOTA: Esta función sigue siendo para la pestaña "Buscar Documentos" (keyword-based).
    # La pestaña "Modo Conversación (IA)" usará find_relevant_chunks para embeddings.
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    search_query = f"%{query}%"
    c.execute('''
        SELECT id, filepath, filename, full_text, doc_type, legal_area, parties, doc_date, summary, keywords, last_updated
        FROM documentos
        WHERE full_text LIKE ? OR filename LIKE ? OR doc_type LIKE ? OR legal_area LIKE ? OR parties LIKE ? OR keywords LIKE ?
        ORDER BY last_updated DESC
        LIMIT ?
    ''', (search_query, search_query, search_query, search_query, search_query, search_query, limit))
    results = c.fetchall()
    conn.close()
    return results

# --- Funciones para la IA (Modo Conversación) ---

# Función para dividir el texto en trozos (chunks)
# Esto es esencial porque los modelos de IA tienen un límite de tokens (cuánta información pueden procesar a la vez)
def get_document_chunks(text, chunk_size=2000, overlap=200):
    chunks = []
    if not text:
        return chunks

    words = text.split()
    current_chunk = []
    current_length = 0

    for word in words:
        # Verifica si añadir la palabra actual excedería el tamaño del chunk
        # +1 es para el espacio entre palabras
        if current_length + len(word) + 1 > chunk_size and current_chunk: # Asegurarse de que current_chunk no esté vacío
            chunks.append(" ".join(current_chunk))
            
            # Para el solapamiento, retrocede un poco
            # Ajuste en el cálculo del solapamiento para evitar división por cero
            num_words_to_overlap = int(overlap / (chunk_size / len(current_chunk) if len(current_chunk) > 0 else 1))
            current_chunk = current_chunk[-num_words_to_overlap:] if current_chunk else []
            
            current_length = sum(len(w) for w in current_chunk) + len(current_chunk) # Recalcular longitud
            
        current_chunk.append(word)
        current_length += len(word) + 1
    
    # Añadir el último chunk si existe
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


# Función para obtener los "embeddings" de texto (representación numérica del significado)
# Utilizaremos el modelo de embedding de Google para esto
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2)) # Reintentar si falla la API
def get_text_embedding(text):
    try:
        # Los embeddings son más eficientes con trozos pequeños, max 1024 tokens ~ 700-800 palabras
        # Por eso get_document_chunks es importante antes de llamar a embeddings masivamente.
        # Aquí truncamos solo para asegurar, pero el chunking previo es la clave.
        text = text.replace("\n", " ") # Normalizar saltos de línea
        if len(text) > 2000: # Truncar si el texto es demasiado largo para el embedding
            text = text[:2000]
        # Asegúrate de usar un modelo de embedding adecuado, 'embedding-001' es común
        return genai.embed_content(model="models/embedding-001", content=text)["embedding"]
    except Exception as e:
        st.warning(f"Error al obtener embedding: {e}. Texto: '{text[:100]}...'")
        return None

# Función para calcular la similitud del coseno entre dos vectores (embeddings) <-- AÑADIDO
def cosine_similarity(vec1, vec2):
    if not isinstance(vec1, np.ndarray):
        vec1 = np.array(vec1)
    if not
