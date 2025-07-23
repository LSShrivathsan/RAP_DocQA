import os
import gradio as gr
import torch
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from PyPDF2 import PdfReader
from docx import Document

embed_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
embed_model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B")

def get_embedding(text):
    inputs = embed_tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = embed_model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        embedding = last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding

chat_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True)
chat_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")

def chunk_text(text, max_tokens=256):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_tokens):
        chunk = " ".join(words[i:i+max_tokens])
        chunks.append(chunk)
    return chunks

class VectorStore:
    def __init__(self, dim=1024):
        self.index = faiss.IndexFlatL2(dim)
        self.data = []

    def add_chunks(self, chunks, filename):
        vectors = [get_embedding(chunk) for chunk in chunks]
        self.index.add(np.array(vectors).astype("float32"))
        for i, chunk in enumerate(chunks):
            self.data.append((filename, i + 1, chunk))

    def search(self, query, top_k=3):
        vector = get_embedding(query).astype("float32").reshape(1, -1)
        D, I = self.index.search(vector, top_k)
        return [self.data[i] for i in I[0] if i < len(self.data)]

vector_store = VectorStore()

def extract_text(file):
    try:
        if file.name.endswith(".pdf"):
            reader = PdfReader(file)
            text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
        elif file.name.endswith(".docx"):
            doc = Document(file)
            text = "\n".join([para.text for para in doc.paragraphs])
        elif file.name.endswith(".txt"):
            text = file.read().decode("utf-8")
        else:
            return ""
        return text.strip()
    except Exception as e:
        return f"Error while extracting text: {str(e)}"

def upload_and_index(files):
    messages = []
    for file in files:
        text = extract_text(file)
        filename = os.path.basename(file.name)
        if not text or text.startswith("Error"):
            messages.append(f"❌ Failed to extract text from {filename}: {text}")
            continue
        try:
            chunks = chunk_text(text)
            if not chunks:
                messages.append(f"❌ No content found in {filename}.")
                continue
            vector_store.add_chunks(chunks, filename)
            messages.append(f"✅ {filename} processed. {len(chunks)} chunks indexed.")
        except Exception as e:
            messages.append(f"❌ Error processing {filename}: {str(e)}")
    return "\n".join(messages)

def answer_question(question):
    try:
        retrieved = vector_store.search(question, top_k=3)
        if not retrieved:
            return "❌ No relevant context found to answer the question."

        context_blocks = []
        sources = []

        for filename, chunk_id, chunk_text in retrieved:
            context_blocks.append(f"[{filename} | Chunk {chunk_id}]\n{chunk_text}")
            sources.append(f"[{filename} | Chunk {chunk_id}]")

        context = "\n\n".join(context_blocks)

        prompt = f"""<|im_start|>system
You are a helpful assistant. Answer questions using the context below:
{context}
<|im_end|>
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant"""

        inputs = chat_tokenizer(prompt, return_tensors="pt").to(chat_model.device)
        outputs = chat_model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7)
        answer = chat_tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

        return f"{answer}\n\n📌 Sources:\n" + "\n".join(sources)
    except Exception as e:
        return f"❌ Error generating answer: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("## 📄 Document Q&A \nUpload multiple documents and ask questions about them.")

    file_input = gr.File(label="Upload Files", file_types=[".pdf", ".docx", ".txt"], file_count="multiple")
    upload_btn = gr.Button("📤 Upload and Index Documents")
    status_output = gr.Textbox(label="Status", lines=4)

    with gr.Row():
        question_input = gr.Textbox(label="Ask a Question")
        ask_btn = gr.Button("💬 Ask")
        answer_output = gr.Textbox(label="Answer", lines=6)

    upload_btn.click(upload_and_index, inputs=file_input, outputs=status_output)
    ask_btn.click(answer_question, inputs=question_input, outputs=answer_output)

demo.launch()
