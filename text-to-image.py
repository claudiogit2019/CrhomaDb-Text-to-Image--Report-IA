import torch
import chromadb
from transformers import CLIPProcessor, CLIPModel
import gradio as gr
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import os
import openai
import PyPDF2
import fitz  # PyMuPDF, para generar vistas previas de PDF
import logging
import re  # Para dividir el texto en secciones

# Configuración de clave de OpenAI (opcional)
openai_api_key = os.getenv("OPENAI_API_KEY", "tu_clave_openai_aqui")
openai.api_key = openai_api_key if openai_api_key != "tu_clave_openai_aqui" else None

# Desactiva el uso de GPU en TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Setup ChromaDB client
client = chromadb.Client()

# Crear o recuperar la colección de documentos médicos
collection_name = "medical_docs_collection"
if collection_name not in client.list_collections():
    collection = client.create_collection(collection_name)
else:
    collection = client.get_collection(collection_name)

# Cargar modelo CLIP
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Diccionario para almacenar los detalles de cada enfermedad junto con sus imágenes
medical_docs = {}
search_history = []  # Historial de búsquedas
MAX_DOCS = 10
DEFAULT_SIMILARITY_THRESHOLD = 0.75  # Umbral de similitud por defecto
SIMILARITY_THRESHOLD = DEFAULT_SIMILARITY_THRESHOLD  # Umbral ajustable

# Configuración básica de logging
logging.basicConfig(level=logging.INFO)

# Funciones de utilidad
def clear_inputs():
    return "", "", None, None  # Valores vacíos para limpiar título, descripción, imágenes y PDF

def clear_search():
    return "", None, None, "", []  # Limpia el campo de texto, imagen de búsqueda y resultados

def calculate_similarity(doc_embedding, query_embedding):
    similarity = cosine_similarity([doc_embedding], [query_embedding])[0][0]
    return similarity

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def generate_pdf_previews(pdf_path):
    previews = []
    with fitz.open(pdf_path) as pdf_document:
        for i in range(len(pdf_document)):
            page = pdf_document.load_page(i)
            pdf_image = page.get_pixmap()
            preview_path = f"pdf_preview_page_{i + 1}.png"
            pdf_image.save(preview_path)
            previews.append(preview_path)
    return previews

def generate_pdf_summary(pdf_text):
    # Extraemos las primeras frases de cada párrafo largo para un resumen mejorado
    sections = re.split(r'\n\n|\r\n\r\n', pdf_text)
    summary = "\n".join(section[:150] + "..." for section in sections if len(section) > 100)[:500]
    return summary

def generate_pdf_embedding(pdf_text):
    pdf_text = pdf_text[:512] if len(pdf_text) > 512 else pdf_text
    pdf_inputs = processor(text=[pdf_text], return_tensors="pt", padding=True, truncation=True, max_length=77)
    with torch.no_grad():
        return model.get_text_features(**pdf_inputs).numpy().flatten().tolist()

# Admin: Subir o actualizar datos de enfermedades con múltiples imágenes y PDF
def upload_disease(title, text, images=None, pdf=None, update=False):
    global medical_docs
    if update and title in medical_docs:
        collection.delete(ids=[title])
        del medical_docs[title]

    if not update and len(medical_docs) >= MAX_DOCS:
        return f"Cannot add more diseases. Maximum of {MAX_DOCS} reached.", update_summary()

    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77)
    with torch.no_grad():
        text_embedding = model.get_text_features(**inputs).numpy().flatten().tolist()

    pdf_embedding = pdf_summary = None
    pdf_previews = []
    if pdf is not None:
        pdf_text = extract_text_from_pdf(pdf)
        pdf_embedding = generate_pdf_embedding(pdf_text)
        pdf_summary = generate_pdf_summary(pdf_text)
        pdf_previews = generate_pdf_previews(pdf)

    image_embeddings = []
    if images is not None:
        try:
            for image_path in images:
                img = Image.open(image_path)
                image_inputs = processor(images=[img], return_tensors="pt", padding=True)
                with torch.no_grad():
                    image_embeddings.append({"embedding": model.get_image_features(**image_inputs).numpy().flatten().tolist(), "file": image_path})
        except Exception as e:
            logging.error(f"Error processing images: {e}")
            return f"Error processing images: {e}", update_summary()

    medical_docs[title] = {
        "text": text,
        "text_embedding": text_embedding,
        "pdf_embedding": pdf_embedding,
        "pdf_summary": pdf_summary,
        "pdf_previews": pdf_previews,
        "images": image_embeddings
    }

    try:
        metadata = {"title": title, "text": text}
        collection.add(embeddings=[text_embedding], metadatas=[metadata], ids=[title])
        return f"Disease '{title}' added successfully!", update_summary()
    except Exception as e:
        logging.error(f"Failed to add disease: {e}")
        return f"Failed to add disease: {e}", update_summary()

def update_summary():
    return "\n".join([f"{i+1}. {title}" for i, title in enumerate(medical_docs.keys())])

def search_medical_docs(query_text=None, query_image=None):
    global SIMILARITY_THRESHOLD
    if not query_text and not query_image:
        return None, "Please enter text or upload an image for search.", "", []

    if query_text:
        inputs = processor(text=[query_text], return_tensors="pt", padding=True, truncation=True, max_length=77)
        with torch.no_grad():
            query_embedding = model.get_text_features(**inputs).numpy().flatten().tolist()
        search_history.append(query_text)  # Añadir al historial de búsqueda
    elif query_image:
        image = Image.open(query_image)
        image_inputs = processor(images=[image], return_tensors="pt", padding=True)
        with torch.no_grad():
            query_embedding = model.get_image_features(**image_inputs).numpy().flatten().tolist()

    try:
        results = collection.query(query_embeddings=[query_embedding], n_results=5)
        response, selected_image, selected_previews = "", None, []
        max_similarity = 0
        for result in results['metadatas'][0]:
            title = result["title"]
            disease = medical_docs.get(title)
            if not disease:
                continue
            text_similarity = calculate_similarity(disease["text_embedding"], query_embedding)
            if disease.get("pdf_embedding"):
                pdf_similarity = calculate_similarity(disease["pdf_embedding"], query_embedding)
                text_similarity = max(text_similarity, pdf_similarity)

            if text_similarity > max_similarity and text_similarity > SIMILARITY_THRESHOLD:
                max_similarity = text_similarity
                response = f"{title}:\n{disease['text']}\n\nPDF Summary:\n{disease['pdf_summary']}\n\n"
                selected_previews = disease["pdf_previews"] + [img["file"] for img in disease["images"]]

        if max_similarity < SIMILARITY_THRESHOLD:
            return "default_pdf_image.png", "No relevant disease found.", "", []
        return selected_previews, response, search_history[-5:], SIMILARITY_THRESHOLD
    except Exception as e:
        logging.error(f"Error querying collection: {e}")
        return None, f"Error querying collection: {e}", "", []

# Interfaz de usuario en Gradio
with gr.Blocks(theme=gr.themes.Soft(), css="""
    .gr-button { color: white; background-color: #1A73E8; border: none; font-weight: bold; }
    .gr-textbox, .gr-markdown { font-family: Arial, sans-serif; font-size: 16px; }
    .gr-image { border-radius: 8px; box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1); }
    body { background-color: #F3F6FB; }
""") as interface:
    gr.Markdown("# 🩺 Text-to-Image Medical Search App")

    with gr.Tab("Admin Panel"):
        admin_title = gr.Textbox(label="Disease Title", placeholder="Enter disease title...")
        admin_text = gr.Textbox(label="Disease Description", placeholder="Enter disease description...")
        admin_images = gr.File(label="Disease Images (upload multiple)", type="filepath", file_count="multiple", elem_id="images")
        admin_pdf = gr.File(label="Disease PDF (optional)", type="filepath", file_count="single", elem_id="pdf")

        upload_btn = gr.Button("Upload Disease")
        update_btn = gr.Button("Update Disease")
        clear_btn = gr.Button("Clear Fields")
        update_checkbox = gr.Checkbox(label="Update existing disease", value=False)

        admin_output = gr.Textbox(label="Upload Status")
        admin_summary = gr.Textbox(label="Uploaded Diseases Summary", interactive=False)

        upload_btn.click(upload_disease, inputs=[admin_title, admin_text, admin_images, admin_pdf, update_checkbox], outputs=[admin_output, admin_summary])
        update_btn.click(upload_disease, inputs=[admin_title, admin_text, admin_images, admin_pdf, update_checkbox], outputs=[admin_output, admin_summary])
        clear_btn.click(clear_inputs, outputs=[admin_title, admin_text, admin_images, admin_pdf])

    # Panel de búsqueda para encontrar enfermedades por texto o imagen
    with gr.Tab("Search Panel"):
        search_query_text = gr.Textbox(label="Enter symptoms or diagnosis", placeholder="Search by symptoms or diagnosis...")
        search_query_image = gr.File(label="Upload an image for search (optional)", type="filepath", file_count="single")
        similarity_slider = gr.Slider(label="Relevance Threshold", minimum=0.5, maximum=0.95, step=0.05, value=DEFAULT_SIMILARITY_THRESHOLD)
        search_btn = gr.Button("Search")
        clear_search_btn = gr.Button("Clear Search")

        output_gallery = gr.Gallery(label="Result Gallery")
        output_text = gr.Textbox(label="Result Details")
        search_history_output = gr.Textbox(label="Recent Searches", interactive=False)

        search_btn.click(
            fn=search_medical_docs, 
            inputs=[search_query_text, search_query_image], 
            outputs=[output_gallery, output_text, search_history_output, similarity_slider]
        )
        similarity_slider.change(lambda x: setattr(globals(), 'SIMILARITY_THRESHOLD', x), inputs=similarity_slider, outputs=None)
        clear_search_btn.click(clear_search, outputs=[search_query_text, search_query_image, output_gallery, output_text, search_history_output])

    interface.launch(share=True)
