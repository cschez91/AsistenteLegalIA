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
import numpy as np # Para cálculos de similitud de vectores
import json # Para serializar/deserializar embeddings a/desde BLOB

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

# Función para calcular la similitud del coseno entre dos vectores (embeddings)
def cosine_similarity(vec1, vec2):
    if not isinstance(vec1, np.ndarray):
        vec1 = np.array(vec1)
    if not isinstance(vec2, np.ndarray):
        vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot_product / (norm_a * norm_b)


# Función para encontrar los chunks más relevantes para una pregunta
def find_relevant_chunks(query, top_k=5):
    query_embedding = get_text_embedding(query)
    if query_embedding is None:
        st.error("No se pudo generar el embedding para la pregunta. No se pueden buscar documentos.")
        return []

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Recuperar todos los chunks y sus embeddings de la base de datos
    c.execute("SELECT chunk_id, document_id, chunk_text, embedding FROM document_chunks")
    all_chunks_data = c.fetchall()
    conn.close()

    similarities = []
    
    # Calcular la similitud de cada chunk con la pregunta
    for chunk_id, document_id, chunk_text, embedding_blob in all_chunks_data:
        try:
            # Decodificar el embedding de BLOB (bytes) a una lista/array de Python
            chunk_embedding = json.loads(embedding_blob.decode('utf-8'))
            similarity = cosine_similarity(query_embedding, chunk_embedding)
            similarities.append((similarity, chunk_id, document_id, chunk_text))
        except Exception as e:
            st.warning(f"Error al procesar embedding del chunk {chunk_id}: {e}")
            continue

    # Ordenar los chunks por su similitud (los más altos primero)
    similarities.sort(key=lambda x: x[0], reverse=True)
    
    relevant_chunks_info = []
    
    # Seleccionar los top_k chunks más relevantes
    for sim, chunk_id, document_id, chunk_text in similarities:
        if len(relevant_chunks_info) >= top_k:
            break
        
        # Obtener el nombre del archivo original al que pertenece este chunk
        conn = sqlite3.connect(DB_FILE)
        doc_cursor = conn.cursor()
        doc_cursor.execute("SELECT filename FROM documentos WHERE id = ?", (document_id,))
        filename = doc_cursor.fetchone()[0]
        conn.close()

        relevant_chunks_info.append({
            "source_filename": filename,
            "source_id": document_id, # ID del documento padre
            "chunk_number": chunk_id, # ID del chunk
            "text": chunk_text
        })
            
    return relevant_chunks_info

# Función para preguntar a Gemini usando el contexto de los documentos
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2)) # Reintentar si falla la API
def ask_gemini(query, context_chunks):
    if not context_chunks:
        return "Lo siento, no encontré información relevante en tus documentos para responder a esa pregunta.", []

    # Construir el prompt con el contexto
    context_text = "\n\n--- Contexto de los documentos ---\n"
    sources_list = []
    
    # Solo añadir fuentes únicas por nombre de archivo
    unique_sources = {} 
    for i, chunk in enumerate(context_chunks):
        # Usar filename y ID para una fuente única
        source_identifier = f"{chunk['source_filename']} (ID: {chunk['source_id']})"
        if source_identifier not in unique_sources:
            unique_sources[source_identifier] = True
            sources_list.append(source_identifier)

        context_text += f"**Documento {chunk['source_filename']} (ID: {chunk['source_id']}, Trozo {chunk['chunk_number']}):**\n"
        context_text += textwrap.dedent(chunk['text']) + "\n\n"
    
    # El prompt instruye a Gemini a usar solo la información proporcionada
    prompt = f"""Eres un asistente legal altamente experto y preciso. Responde a la pregunta del usuario utilizando ÚNICAMENTE la información proporcionada en el siguiente contexto de documentos legales. Si la pregunta no puede ser respondida con la información del contexto proporcionado, responde con "No tengo información suficiente en los documentos proporcionados para responder a esa pregunta." No inventes información.

    {context_text}

    ---

    Pregunta del usuario: {query}

    Respuesta:"""

    try:
        response = model.generate_content(prompt)
        # Acceder al texto de la respuesta. Algunos resultados pueden no tener 'text' si son bloqueados por seguridad, etc.
        if hasattr(response, '_result') and response._result.candidates:
            first_candidate = response._result.candidates[0]
            if hasattr(first_candidate, 'content') and hasattr(first_candidate.content, 'parts'):
                response_text = first_candidate.content.parts[0].text
                return response_text, sources_list
        return "Hubo un problema al generar la respuesta. Por favor, intenta de nuevo.", sources_list
    except Exception as e:
        st.error(f"Error al comunicarse con la API de Gemini: {e}. Esto puede ser un problema de cuota, clave incorrecta o problema de red. Por favor, revisa tu clave API y tu conexión.")
        return "Lo siento, no pude comunicarme con la IA. Por favor, intenta de nuevo más tarde o verifica tu configuración API.", sources_list


# --- Interfaz de Usuario con Streamlit ---

st.set_page_config(layout="wide", page_title="Asistente Legal")
st.title("👨‍⚖️ Mi Asistente Personal de Documentos Legales")

# Inicializar la base de datos al inicio de la aplicación
init_db()

# Navegación entre secciones
tab1, tab2, tab3, tab4 = st.tabs(["📂 Cargar Documentos", "🔍 Buscar Documentos", "💬 Modo Conversación (IA)", "📊 Gestión y Estadísticas"])

with tab1:
    st.header("Cargar Nuevos Documentos")
    st.write("Puedes arrastrar y soltar archivos .docx o .pdf individualmente o especificar una carpeta para carga masiva.")

    uploaded_files = st.file_uploader("Arrastra y suelta archivos .docx o .pdf aquí o haz clic para seleccionar",
                                     type=["docx", "pdf"], accept_multiple_files=True)

    if uploaded_files:
        st.subheader("Documentos Seleccionados para Carga Individual:")
        if st.button("Procesar y Añadir Documentos Seleccionados"):
            for uploaded_file in uploaded_files:
                temp_filepath = os.path.join(TARGET_DIR, uploaded_file.name)

                try:
                    with open(temp_filepath, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                except Exception as e:
                    st.error(f"Error al guardar el archivo '{uploaded_file.name}': {e}")
                    continue 

                full_text = None
                if uploaded_file.name.lower().endswith(".docx"):
                    full_text = get_docx_text(temp_filepath)
                elif uploaded_file.name.lower().endswith(".pdf"):
                    full_text = get_pdf_text(temp_filepath)
                else:
                    st.error(f"Tipo de archivo no soportado para {uploaded_file.name}. Solo .docx y .pdf son aceptados.")
                    if os.path.exists(temp_filepath):
                        os.remove(temp_filepath)
                    continue

                if full_text is not None:
                    insert_document(
                        filepath=temp_filepath,
                        filename=uploaded_file.name,
                        full_text=full_text
                    )
                else:
                    st.error(f"No se pudo extraer texto de {uploaded_file.name}. Asegúrate de que sea un archivo válido.")
                    if os.path.exists(temp_filepath):
                        os.remove(temp_filepath)
            st.success("Proceso de carga individual finalizado.")


    st.markdown("---")
    st.subheader("Carga Masiva desde Carpeta Local (Para tus documentos existentes)")
    st.info("Ingresa la ruta completa de la carpeta que contiene tus documentos .docx y .pdf.")
    st.warning("⚠️ **¡Importante!** La carga masiva desde una carpeta local solo funciona cuando ejecutas la aplicación en tu **computadora local**. En Streamlit Cloud, la aplicación no tiene acceso a tus archivos locales. Usa 'Arrastra y suelta' arriba para cargar documentos en la nube.") 

    local_folder_path = st.text_input("Ruta completa de la carpeta con tus documentos (.docx/.pdf):", value=r"C:\Users\tu_usuario\Documents\MisEscritosLegales")
    st.caption("Ejemplo para Windows: `C:\\Users\\TuUsuario\\Documents\\MisEscritosLegales` o para macOS/Linux: `/Users/TuUsuario/Documents/MisEscritosLegales`")

    if st.button("Iniciar Carga Masiva"):
        # Asegúrate de que la ruta existe y no estamos en un entorno de Streamlit Cloud intentando acceder a una ruta local
        if os.path.isdir(local_folder_path) and os.path.isabs(local_folder_path) and "STREAMLIT_SERVER_URL" not in os.environ: 
            st.info("Iniciando carga masiva desde la carpeta local...") 
            file_count = 0
            with st.spinner("Cargando documentos... Esto puede tardar un tiempo si tienes muchos archivos."):
                for root, _, files in os.walk(local_folder_path):
                    for filename in files:
                        filepath = os.path.join(root, filename)
                        full_text = None
                        
                        if filename.lower().endswith('.docx'):
                            full_text = get_docx_text(filepath)
                        elif filename.lower().endswith('.pdf'):
                            full_text = get_pdf_text(filepath)
                        else:
                            continue

                        if full_text is not None:
                            if insert_document(filepath=filepath, filename=filename, full_text=full_text):
                                file_count += 1
                        else:
                            st.warning(f"Omitiendo {filename} (no se pudo extraer texto o es un archivo corrupto).")
                st.success(f"Carga masiva finalizada. Se procesaron {file_count} nuevos documentos (los existentes se omitieron o tuvieron errores de extracción).")
            st.info("La base de datos ya está lista para búsquedas.")
        else:
            st.error("La ruta de carpeta ingresada no es válida o no es una ruta local accesible. Recuerda que la carga masiva directa solo funciona localmente y no en Streamlit Cloud.")


with tab2:
    st.header("🔍 Buscar Documentos")
    st.write("Ingresa palabras clave o frases para encontrar documentos relevantes.")
    search_query = st.text_input("Ingresa tu consulta:", placeholder="Ej: contrato de arrendamiento, sentencia de divorcio")

    if st.button("Buscar en Documentos"):
        if search_query:
            results = search_documents(search_query)
            if results:
                st.subheader(f"Resultados de la Búsqueda para '{search_query}':")
                df_results = pd.DataFrame(results, columns=["ID", "Ruta Archivo", "Nombre Archivo", "Texto Completo",
                                                            "Tipo", "Área Legal", "Partes", "Fecha Doc", "Resumen",
                                                            "Palabras Clave", "Última Actualización"])
                df_results_display = df_results.drop(columns=["Texto Completo"])

                st.dataframe(df_results_display, use_container_width=True)

                st.markdown("---")
                st.subheader("Detalles de los Documentos Encontrados:")
                for i, row_data in enumerate(results):
                    doc_id, filepath_result, filename, full_text, doc_type, legal_area, parties, doc_date, summary, keywords, last_updated = row_data
                    
                    with st.expander(f"**{i+1}. {filename}** (ID: {doc_id})"):
                        st.write(f"**Tipo de Documento:** {doc_type or 'N/A'}")
                        st.write(f"**Área Legal:** {legal_area or 'N/A'}")
                        st.write(f"**Partes:** {parties or 'N/A'}")
                        st.write(f"**Fecha del Documento:** {doc_date or 'N/A'}")
                        st.write(f"**Palabras Clave:** {keywords or 'N/A'}")
                        st.write(f"**Última Actualización del Registro:** {last_updated.split('T')[0]}")
                        st.write(f"**Ruta del Archivo Original:** `{filepath_result}`")

                        if st.checkbox(f"Mostrar texto completo de '{filename}'", key=f"show_text_{doc_id}"):
                            st.text_area("Texto Completo:", full_text, height=300, key=f"full_text_area_{doc_id}")
                        
            else:
                st.info("No se encontraron documentos que coincidan con tu búsqueda. Intenta con otras palabras clave.")
        else:
            st.warning("Por favor, ingresa una consulta para buscar.")

    st.markdown("---")
    st.subheader("🚀 Funcionalidades Avanzadas (Próximamente):")
    st.info("""
    Aquí se integrarán las funcionalidades de IA para:
    * **Modo Conversación (Q&A):** Haz preguntas complejas sobre el contenido de tus documentos y obtén respuestas directas.
    * **Asistencia en Redacción:** Genera borradores o cláusulas basadas en tus propios documentos y parámetros.
    """)

with tab3: # NUEVA PESTAÑA: Modo Conversación (IA)
    st.header("💬 Modo Conversación (IA)")
    st.write("Haz preguntas sobre el contenido de tus documentos indexados (leyes, sentencias, contratos, etc.) y la IA te responderá basándose en esa información.")

    qa_query = st.text_area("Tu pregunta para la IA (ej. '¿Cuáles son las obligaciones del arrendatario en el contrato de la calle X?', 'Resume la sentencia ID 123'):", height=100)

    if st.button("Preguntar a la IA sobre mis documentos"):
        if qa_query:
            with st.spinner("Buscando en tus documentos y generando respuesta con IA..."):
                # PASO CRÍTICO: Usar find_relevant_chunks basado en embeddings
                context_chunks = find_relevant_chunks(qa_query, top_k=5) # Ajusta top_k si quieres más o menos contexto
                
                if context_chunks:
                    # Llamar a la función que interactúa con Gemini
                    response_text, sources = ask_gemini(qa_query, context_chunks)
                    
                    st.subheader("Respuesta de la IA:")
                    st.markdown(response_text)
                    
                    if sources:
                        st.subheader("Fuentes consultadas en tus documentos:")
                        for source in sources:
                            st.write(f"- {source}")
                    else:
                        st.info("No se encontraron fuentes específicas para esta respuesta.")
                else:
                    st.info("No se encontraron documentos relevantes en tu biblioteca para responder a esa pregunta. Intenta con una consulta diferente o carga más documentos.")
        else:
            st.warning("Por favor, ingresa tu pregunta para la IA.")

with tab4: # PESTAÑA RENOMBRADA a tab4 para hacer espacio a la de conversación
    st.header("📊 Gestión y Estadísticas de la Base de Datos")
    st.write("Herramientas para monitorear y mantener tu asistente.")

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM documentos")
    total_docs = c.fetchone()[0]
    
    # Obtener el número de chunks para mostrar cuánta información hay en la DB de embeddings
    c.execute("SELECT COUNT(*) FROM document_chunks")
    total_chunks = c.fetchone()[0]
    conn.close()

    st.metric(label="Total de Documentos Indexados", value=total_docs)
    st.metric(label="Total de Fragmentos (Chunks) con Embeddings", value=total_chunks)

    st.write(f"La base de datos se guarda en el archivo: `{os.path.join(os.getcwd(), DB_FILE)}`")
    st.write(f"Los documentos originales se guardan en la carpeta: `{os.path.join(os.getcwd(), TARGET_DIR)}`")


    st.subheader("Ver y Editar Documentos (Funcionalidad Avanzada)")
    st.info("Esta sección permitirá explorar y editar los metadatos de los documentos. Actualmente, muestra los últimos 20.")
    
    conn = sqlite3.connect(DB_FILE)
    df_all_docs = pd.read_sql_query("SELECT id, filename, doc_type, legal_area, parties, doc_date, last_updated FROM documentos ORDER BY last_updated DESC LIMIT 20", conn)
    conn.close()
    
    st.dataframe(df_all_docs, use_container_width=True)

    st.subheader("Mantenimiento de la Base de Datos")
    st.warning("Cuidado: Estas acciones son permanentes y no se pueden deshacer fácilmente.")
    
    if st.button("Limpiar Base de Datos (Eliminar TODA la Información Indexada)"):
        st.error("¡Advertencia! Esta acción eliminará permanentemente todos los registros de documentos de la base de datos (tanto de la tabla `documentos` como de `document_chunks`).")
        st.info("Tus archivos .docx y .pdf originales en la carpeta `documentos_originales` NO se borrarán. Solo se eliminará su índice de la base de datos.")
        
        col_clear1, col_clear2 = st.columns(2)
        with col_clear1:
            if st.button("CONFIRMAR: Eliminar TODO de la base de datos", key="confirm_clear_db_final"):
                conn = sqlite3.connect(DB_FILE)
                c = conn.cursor()
                c.execute("DELETE FROM document_chunks")
                c.execute("DELETE FROM documentos")
                conn.commit()
                conn.close()
                st.success("Base de datos limpia. Recarga la página para ver los cambios o sube nuevos documentos.")
                st.experimental_rerun() # Recarga la app para reflejar los cambios
            with col_clear2:
                st.info("Haz clic en el botón de confirmación si estás seguro.")