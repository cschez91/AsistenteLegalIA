import streamlit as st
import sqlite3
import os
from docx import Document
from PyPDF2 import PdfReader
from datetime import datetime
import pandas as pd
import textwrap
from tenacity import retry, stop_after_attempt, wait_fixed
import numpy as np
import json

# --- Configuración de la base de datos y IA ---
DB_FILE = 'documentos_legales.db'
TARGET_DIR = "documentos_originales"

# Asegurarse de que la carpeta de destino exista al inicio de la aplicación
os.makedirs(TARGET_DIR, exist_ok=True)

# Variables globales para LLM y Embeddings
llm = None
embeddings_model = None
llm_provider = None
api_key_llm = None # Para almacenar la clave que realmente se usó

# --- INICIALIZACIÓN CONDICIONAL DE LLM Y EMBEDDINGS (PRIORIZANDO COHERE) ---
# Intenta cargar la clave de Cohere primero
cohere_api_key = os.getenv("COHERE_API_KEY")

if cohere_api_key:
    llm_provider = "Cohere"
    api_key_llm = cohere_api_key
    try:
        from langchain_cohere import ChatCohere, CohereEmbeddings
        llm = ChatCohere(model="command", temperature=0.3, cohere_api_key=cohere_api_key)
        embeddings_model = CohereEmbeddings(model="embed-multilingual-v3.0", cohere_api_key=cohere_api_key)
        st.success("Usando Cohere como proveedor de LLM y Embeddings.", icon="✅")
    except ImportError:
        st.error("Error: No se pudieron importar los módulos de Cohere. Asegúrate de que 'langchain-cohere' esté instalado en requirements.txt.", icon="❌")
        llm = None
        embeddings_model = None
        llm_provider = None
    except Exception as e:
        st.error(f"Error al inicializar modelos de Cohere: {e}. Asegúrate de que COHERE_API_KEY sea válida.", icon="❌")
        llm = None
        embeddings_model = None
        llm_provider = None
else:
    # Si la clave de Cohere no está presente, intenta con Google Gemini
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if google_api_key:
        llm_provider = "Google Gemini"
        api_key_llm = google_api_key
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
            llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key, temperature=0.3)
            embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
            st.info("Usando Google Gemini como proveedor de LLM y Embeddings (Cohere no disponible).", icon="ℹ️")
        except ImportError:
            st.error("Error: No se pudieron importar los módulos de Google Gemini. Asegúrate de que 'langchain-google-genai' esté instalado en requirements.txt.", icon="❌")
            llm = None
            embeddings_model = None
            llm_provider = None
        except Exception as e:
            st.error(f"Error al inicializar modelos de Google Gemini: {e}. Asegúrate de que GOOGLE_API_KEY sea válida.", icon="❌")
            llm = None
            embeddings_model = None
            llm_provider = None
    else:
        st.warning("¡ATENCIÓN! No se encontró ninguna API Key para Cohere o Google Gemini. Algunas funcionalidades estarán limitadas.", icon="⚠️")
        llm = None
        embeddings_model = None
        llm_provider = None

# Verificar si se pudo inicializar algún LLM y Embedding. Si no, detener la aplicación.
if llm is None or embeddings_model is None:
    st.error("No se pudo configurar un modelo de lenguaje (LLM) o de embeddings. Por favor, revisa tus API Keys en las variables de entorno de Streamlit Cloud.", icon="❗")
    st.stop() # Detener la ejecución si no hay LLM o embeddings disponibles


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
    # NUEVA TABLA: document_chunks
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

        # Generar chunks y embeddings para el nuevo documento
        chunks = get_document_chunks(full_text_to_insert)
        st.info(f"Generando embeddings para {len(chunks)} trozos de '{filename}' con {llm_provider} Embeddings...")
        for i, chunk_text in enumerate(chunks):
            embedding = get_text_embedding(chunk_text) # Esta función ahora usará el modelo configurado
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
def get_document_chunks(text, chunk_size=2000, overlap=200):
    chunks = []
    if not text:
        return chunks
    
    # Simple tokenization by words
    words = text.split()
    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 > chunk_size and current_chunk: # +1 for space
            chunks.append(" ".join(current_chunk))
            # Start new chunk with overlap
            overlap_words_count = min(len(current_chunk), overlap)
            current_chunk = current_chunk[-overlap_words_count:]
            current_length = sum(len(w) for w in current_chunk) + (len(current_chunk) - 1 if len(current_chunk) > 0 else 0)
        
        current_chunk.append(word)
        current_length += len(word) + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

# Función para generar embeddings
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def get_text_embedding(text):
    if embeddings_model is None:
        st.error("Modelo de embeddings no inicializado. No se pueden generar embeddings.")
        return None
    try:
        # La función de CohereEmbeddings devuelve una lista de embeddings (aunque sea uno solo)
        # Asegúrate de tomar el primer elemento si solo esperas un embedding
        response = embeddings_model.embed_documents([text])
        if response and isinstance(response, list) and len(response) > 0:
            return response[0] # Retorna el primer embedding
        else:
            st.error(f"El modelo de embeddings no retornó un embedding válido para el texto. Tipo: {type(response)}, Valor: {response}")
            return None
    except Exception as e:
        st.error(f"Error al obtener embedding: {e}")
        raise # Vuelve a lanzar la excepción para que tenacity pueda reintentar

# Función para buscar fragmentos relevantes en la DB usando embeddings
def find_relevant_chunks(query_text, top_k=5):
    if embeddings_model is None:
        st.error("Modelo de embeddings no inicializado. No se puede buscar por relevancia.")
        return []

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Generar el embedding de la consulta
    query_embedding = get_text_embedding(query_text)
    if query_embedding is None:
        conn.close()
        return []

    # Recuperar todos los chunks con embeddings
    c.execute("SELECT document_id, chunk_text, embedding FROM document_chunks")
    all_chunks = c.fetchall()
    conn.close()

    similarities = []
    for doc_id, chunk_text, embedding_blob in all_chunks:
        try:
            # Deserializar el embedding de BLOB a lista/array
            stored_embedding = json.loads(embedding_blob.decode('utf-8'))
            
            # Calcular similitud del coseno
            # Asegurarse de que ambos embeddings sean arrays de numpy y que sean del mismo tamaño
            vec1 = np.array(query_embedding)
            vec2 = np.array(stored_embedding)

            if vec1.shape != vec2.shape:
                st.warning(f"Dimensiones de embedding no coinciden. Consulta: {vec1.shape}, Almacenado: {vec2.shape}. Saltando chunk.")
                continue

            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            similarities.append((similarity, doc_id, chunk_text))
        except Exception as e:
            st.warning(f"Error al procesar chunk o embedding: {e}")
            continue

    similarities.sort(key=lambda x: x[0], reverse=True) # Ordenar por similitud
    
    relevant_chunks = []
    seen_document_ids = set() # Para evitar duplicados de documentos principales

    for sim, doc_id, chunk_text in similarities:
        if sim > 0.60: # Umbral de similitud (ajustable)
            relevant_chunks.append({"document_id": doc_id, "chunk_text": chunk_text, "similarity": sim})
            # Opcional: limitar chunks por documento o un total máximo
            if len(relevant_chunks) >= top_k:
                break
    return relevant_chunks


# --- Interfaz de Usuario de Streamlit ---
st.set_page_config(layout="wide", page_title="Asistente Legal IA")

# Inicializar la base de datos si no existe
init_db()

st.title("Asistente Legal IA ⚖️")

# Pestañas
tab_cargar, tab_buscar, tab_conversar, tab_gestion = st.tabs([
    "📂 Cargar Documentos", 
    "🔍 Buscar Documentos", 
    "💬 Modo Conversación (IA)",
    "📊 Gestión y Estadísticas"
])


with tab_cargar:
    st.header("Cargar Nuevos Documentos Legales")
    uploaded_files = st.file_uploader(
        "Arrastra y suelta archivos .docx o .pdf aquí o haz clic para seleccionar",
        type=["docx", "pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        progress_bar = st.progress(0)
        for i, uploaded_file in enumerate(uploaded_files):
            # Guardar archivo temporalmente
            file_path = os.path.join(TARGET_DIR, uploaded_file.name)
            try:
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                full_text = None
                if uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    full_text = get_docx_text(file_path)
                    doc_type = "DOCX"
                elif uploaded_file.type == "application/pdf":
                    full_text = get_pdf_text(file_path)
                    doc_type = "PDF"
                
                if full_text:
                    st.write(f"Procesando: **{uploaded_file.name}**")
                    # La sección para extracción de metadatos con IA está COMENTADA.
                    # Si quieres usarla, DESCOMENTA cada una de las líneas, eliminando el '#'.
                    # Ten en cuenta que esto requiere que el LLM esté inicializado.
                    # if llm and st.checkbox(f"Extraer metadatos con IA para {uploaded_file.name} (experimental)"):
                    #     with st.spinner("Extrayendo metadatos con IA..."):
                    #         try:
                    #             short_text = full_text[:4000] 
                    #             prompt_metadata = f"""Extrae los siguientes metadatos del siguiente documento legal:
                    #             - Tipo de Documento (ej. Contrato, Sentencia, Ley, Demanda, etc.)
                    #             - Área Legal (ej. Civil, Penal, Mercantil, Laboral, etc.)
                    #             - Partes Involucradas (nombres de personas o entidades)
                    #             - Fecha del Documento (formato AAAA-MM-DD si es posible)
                    #             - Resumen breve (2-3 líneas)
                    #             - Palabras Clave (3-5 relevantes)

                    #             Documento:
                    #             {short_text}

                    #             Formato de salida (JSON):
                    #             ```json
                    #             {{
                    #                 "tipo_documento": "...",
                    #                 "area_legal": "...",
                    #                 "partes": "...",
                    #                 "fecha_documento": "AAAA-MM-DD",
                    #                 "resumen": "...",
                    #                 "palabras_clave": "..."
                    #             }}
                    #             ```
                    #             """
                    #             response = llm.invoke(prompt_metadata)
                    #             extracted_metadata = json.loads(response.content.strip("```json\n```").strip())
                    #             
                    #             doc_type_ai = extracted_metadata.get("tipo_documento", "")
                    #             legal_area_ai = extracted_metadata.get("area_legal", "")
                    #             parties_ai = extracted_metadata.get("partes", "")
                    #             doc_date_ai = extracted_metadata.get("fecha_documento", "")
                    #             summary_ai = extracted_metadata.get("resumen", "")
                    #             keywords_ai = extracted_metadata.get("palabras_clave", "")
                    #             
                    #             st.json(extracted_metadata)
                    #         except Exception as ai_e:
                    #             st.warning(f"No se pudieron extraer metadatos con IA para {uploaded_file.name}: {ai_e}")
                    #             doc_type_ai, legal_area_ai, parties_ai, doc_date_ai, summary_ai, keywords_ai = "", "", "", "", "", ""
                    else:
                        # Si la extracción con IA no está activa o falla, usa valores por defecto
                        doc_type_ai, legal_area_ai, parties_ai, doc_date_ai, summary_ai, keywords_ai = "", "", "", "", "", ""

                    insert_document(
                        file_path, 
                        uploaded_file.name, 
                        full_text,
                        doc_type=doc_type_ai or doc_type, # Usar el tipo detectado o el de IA
                        legal_area=legal_area_ai, 
                        parties=parties_ai, 
                        doc_date=doc_date_ai, 
                        summary=summary_ai, 
                        keywords=keywords_ai
                    )
                else:
                    st.error(f"No se pudo extraer texto de '{uploaded_file.name}'.")
            except Exception as e:
                st.error(f"Error al guardar o procesar '{uploaded_file.name}': {e}")
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        st.success("Todos los archivos procesados.")
        st.experimental_rerun() # Refrescar para mostrar documentos actualizados


with tab_buscar:
    st.header("Buscar Documentos Legales")
    search_query = st.text_input("Introduce tu consulta (ej. 'contrato de arrendamiento', 'sentencia divorcio')")

    if search_query:
        st.subheader("Resultados de la Búsqueda")
        results = search_documents(search_query)

        if results:
            df_results = pd.DataFrame(results, columns=[
                "ID", "Ruta de Archivo", "Nombre de Archivo", "Texto Completo", "Tipo", "Área Legal", 
                "Partes", "Fecha Documento", "Resumen", "Palabras Clave", "Última Actualización"
            ])
            # Eliminar la columna de texto completo si es muy grande para mostrar
            df_results = df_results.drop(columns=["Texto Completo"]) 
            st.dataframe(df_results, use_container_width=True)

            # Opción para ver el texto completo de un documento
            doc_id_to_view = st.number_input("Introduce el ID del documento para ver su texto completo:", min_value=1, format="%d")
            if st.button("Ver Texto Completo") and doc_id_to_view:
                conn = sqlite3.connect(DB_FILE)
                c = conn.cursor()
                c.execute("SELECT full_text FROM documentos WHERE id = ?", (doc_id_to_view,))
                full_text = c.fetchone()
                conn.close()
                if full_text:
                    st.text_area(f"Texto Completo del Documento ID {doc_id_to_view}", full_text[0], height=500)
                else:
                    st.warning("Documento no encontrado.")
        else:
            st.info("No se encontraron documentos que coincidan con tu búsqueda.")


with tab_conversar:
    st.header("Modo Conversación con IA")
    st.info(f"Usando {llm_provider} para el modo conversación.")

    user_question = st.text_input("Hazle una pregunta a tus documentos (ej. '¿Cuáles son los derechos de un arrendatario?', 'Resume las obligaciones del demandado en el caso X')")

    if user_question:
        with st.spinner("Buscando información relevante y generando respuesta..."):
            relevant_chunks = find_relevant_chunks(user_question)
            
            if relevant_chunks:
                context = "\n\n".join([chunk["chunk_text"] for chunk in relevant_chunks])
                
                # Construye el prompt para el LLM
                prompt = f"""Eres un asistente legal experto. Utiliza únicamente la información proporcionada a continuación para responder a la pregunta. 
                Si la información proporcionada no es suficiente para responder la pregunta, di 'No tengo suficiente información en los documentos para responder a esa pregunta.'.
                No inventes información. Responde de manera concisa y profesional.

                Información relevante de los documentos:
                {context}

                Pregunta del usuario:
                {user_question}

                Respuesta:
                """
                try:
                    response = llm.invoke(prompt)
                    st.markdown("---")
                    st.subheader("Respuesta del Asistente IA:")
                    st.write(response.content)

                    st.markdown("---")
                    st.subheader("Fragmentos relevantes utilizados:")
                    for i, chunk in enumerate(relevant_chunks):
                        st.expander(f"Fragmento {i+1} (Similitud: {chunk['similarity']:.2f}) - Documento ID: {chunk['document_id']}")
                        st.text(chunk['chunk_text'])
                except Exception as e:
                    st.error(f"Error al generar respuesta con la IA: {e}")
            else:
                st.info("No se encontraron fragmentos relevantes en los documentos para tu pregunta. Intenta reformularla.")


with tab_gestion:
    st.header("Gestión y Estadísticas de Documentos")

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # Contar documentos
    c.execute("SELECT COUNT(*) FROM documentos")
    total_docs = c.fetchone()[0]
    st.metric("Total de Documentos Indexados", total_docs)

    # Contar fragmentos con embeddings
    c.execute("SELECT COUNT(*) FROM document_chunks")
    total_chunks = c.fetchone()[0]
    st.metric("Total de Fragmentos (Chunks) con Embeddings", total_chunks)

    # Mostrar todos los documentos en la base de datos
    st.subheader("Documentos en la Base de Datos")
    c.execute("SELECT id, filename, doc_type, legal_area, parties, doc_date, last_updated FROM documentos ORDER BY last_updated DESC")
    all_docs = c.fetchall()
    conn.close()

    if all_docs:
        df_all_docs = pd.DataFrame(all_docs, columns=[
            "ID", "Nombre de Archivo", "Tipo", "Área Legal", "Partes", "Fecha Documento", "Última Actualización"
        ])
        st.dataframe(df_all_docs, use_container_width=True)
    else:
        st.info("No hay documentos en la base de datos.")

    st.markdown("---")
    st.subheader("Mantenimiento de la Base de Datos")

    # Botón para limpiar la base de datos
    if st.button("Limpiar Base de Datos (Eliminar TODA la Información Indexada)", type="secondary"):
        confirm_delete = st.checkbox("Confirmo que deseo eliminar **todos** los documentos y sus fragmentos de la base de datos.")
        if confirm_delete:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            try:
                c.execute("DELETE FROM document_chunks") # Eliminar chunks primero por la FK
                c.execute("DELETE FROM documentos")
                conn.commit()
                st.success("¡Base de datos limpiada con éxito! Todos los documentos indexados han sido eliminados.")
                st.warning("Deberás volver a cargar tus documentos para re-indexarlos.")
                st.experimental_rerun() # Refrescar la app para actualizar los contadores
            except Exception as e:
                st.error(f"Error al limpiar la base de datos: {e}")
            finally:
                conn.close()
        else:
            st.info("Confirma la eliminación para proceder.")

    # Opción para eliminar un documento específico por ID
    st.markdown("---")
    st.subheader("Eliminar Documento Específico")
    doc_id_to_delete = st.number_input("Introduce el ID del documento a eliminar:", min_value=1, format="%d", key="delete_specific_doc")
    if st.button("Eliminar Documento por ID", type="secondary"):
        if doc_id_to_delete:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            try:
                c.execute("SELECT filepath, filename FROM documentos WHERE id = ?", (doc_id_to_delete,))
                doc_info = c.fetchone()
                if doc_info:
                    filepath_to_delete = doc_info[0]
                    filename_to_delete = doc_info[1]

                    # Eliminar chunks asociados
                    c.execute("DELETE FROM document_chunks WHERE document_id = ?", (doc_id_to_delete,))
                    # Eliminar el documento de la tabla 'documentos'
                    c.execute("DELETE FROM documentos WHERE id = ?", (doc_id_to_delete,))
                    conn.commit()

                    # Eliminar el archivo físico del sistema de archivos
                    if os.path.exists(filepath_to_delete):
                        os.remove(filepath_to_delete)
                        st.success(f"Documento '{filename_to_delete}' (ID: {doc_id_to_delete}) y sus fragmentos eliminados de la base de datos y el archivo físico.")
                    else:
                        st.warning(f"Documento '{filename_to_delete}' (ID: {doc_id_to_delete}) y sus fragmentos eliminados de la base de datos, pero el archivo físico no fue encontrado en '{filepath_to_delete}'.")
                    
                    st.experimental_rerun() # Refrescar la app para actualizar los contadores
                else:
                    st.warning("Documento no encontrado con el ID proporcionado.")
            except Exception as e:
                st.error(f"Error al eliminar el documento: {e}")
            finally:
                conn.close()
        else:
            st.warning("Por favor, introduce un ID de documento válido para eliminar.")

# Asegurarse de inicializar la DB al inicio de la app
init_db()