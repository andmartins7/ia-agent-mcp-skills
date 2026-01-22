import os
import shutil
from pathlib import Path
from typing import List
from mcp.server.fastmcp import FastMCP

# Extratores
from pypdf import PdfReader
from bs4 import BeautifulSoup

# Vector Store & Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# --- CONFIGURAÇÃO (SANDBOX) ---
TARGET_DIRECTORY = Path("./dados_processos")
TARGET_DIRECTORY.mkdir(parents=True, exist_ok=True)
CHROMA_PATH = "./chroma_db_store" # Pasta onde o banco vetorial será salvo

# Inicializa o Servidor
mcp = FastMCP("Universal Document Processor + RAG")

# Modelo local, leve e eficiente para CPU
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Inicializa o Banco Vetorial (Persistente)
vector_db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embedding_function,
    collection_name="processos_juridicos"
)

# --- FUNÇÕES AUXILIARES DE LEITURA (MANTIDAS) ---
def extract_text_raw(file_path: Path) -> str:
    """Extrai texto bruto dependendo da extensão."""
    try:
        suffix = file_path.suffix.lower()
        if suffix == '.pdf':
            reader = PdfReader(file_path)
            text = []
            for i, page in enumerate(reader.pages):
                txt = page.extract_text()
                if txt: text.append(txt)
            return "\n".join(text)
        elif suffix in ['.html', '.htm']:
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f, 'html.parser')
                for s in soup(["script", "style"]): s.decompose()
                return soup.get_text(separator='\n')
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    except Exception as e:
        return ""

# --- FERRAMENTAS EXISTENTES (SIMPLIFICADAS) ---

@mcp.tool()
def list_available_files() -> List[str]:
    """Lista arquivos na pasta."""
    try:
        return [f.name for f in TARGET_DIRECTORY.iterdir() if f.is_file()]
    except: return []

@mcp.tool()
def read_file_content(filename: str) -> str:
    """Lê o arquivo INTEIRO (Use apenas para arquivos pequenos ou resumos)."""
    path = (TARGET_DIRECTORY / filename).resolve()
    if not path.exists(): return "Arquivo não encontrado."
    return extract_text_raw(path)

@mcp.tool()
def save_document(filename: str, content: str) -> str:
    """Salva um novo documento no disco."""
    try:
        with open(TARGET_DIRECTORY / filename, "w", encoding="utf-8") as f:
            f.write(content)
        return "Salvo com sucesso."
    except Exception as e: return f"Erro: {e}"

# --- NOVAS FERRAMENTAS DE RAG (VECTOR SEARCH) ---

@mcp.tool()
def index_document(filename: str) -> str:
    """
    IMPORTANTE: Executa a indexação vetorial de um arquivo.
    Use isso ANTES de tentar pesquisar trechos nele.
    Isso 'lê' o arquivo, quebra em pedaços e salva na memória de busca.
    """
    file_path = (TARGET_DIRECTORY / filename).resolve()
    if not file_path.exists(): return "Arquivo não encontrado."

    # 1. Extrair Texto
    raw_text = extract_text_raw(file_path)
    if not raw_text: return "Não foi possível extrair texto ou arquivo vazio."

    # 2. Quebrar em Chunks (Pedaços)
    # Chunk size de 1000 caracteres com overlap de 200 é ideal para contexto jurídico
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.create_documents([raw_text], metadatas=[{"source": filename}])

    # 3. Inserir no ChromaDB
    # Isso converte texto em números (vetores) e salva no disco
    vector_db.add_documents(chunks)
    
    return f"Sucesso: Arquivo '{filename}' indexado. Gerados {len(chunks)} fragmentos pesquisáveis."

@mcp.tool()
def search_knowledge_base(query: str, k: int = 4) -> str:
    """
    Pesquisa SEMÂNTICA no banco de dados.
    Use para encontrar fatos específicos sem ler o arquivo todo.
    Ex: query="O que a testemunha João disse sobre o vazamento?"
    k: número de trechos para retornar (padrão 4).
    """
    print(f"--- 🔎 Buscando por: '{query}' ---")
    results = vector_db.similarity_search(query, k=k)
    
    output = "--- RESULTADOS DA BUSCA RELEVANTES ---\n"
    for i, res in enumerate(results):
        source = res.metadata.get("source", "desconhecido")
        output += f"\n[Trecho {i+1} | Fonte: {source}]:\n...{res.page_content}...\n"
    
    return output

if __name__ == "__main__":
    mcp.run()