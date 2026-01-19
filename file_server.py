import os
from pathlib import Path
from typing import List
from mcp.server.fastmcp import FastMCP

# Bibliotecas de Extração
from pypdf import PdfReader
from bs4 import BeautifulSoup

# --- CONFIGURAÇÃO (SANDBOX) ---
TARGET_DIRECTORY = Path("./dados_processos")
TARGET_DIRECTORY.mkdir(parents=True, exist_ok=True)

# Inicializa o Servidor
mcp = FastMCP("Universal Document Processor")

# --- FUNÇÕES AUXILIARES DE LEITURA ---
def extract_text_from_pdf(file_path: Path) -> str:
    try:
        reader = PdfReader(file_path)
        text = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text.append(f"--- PÁGINA {i+1} ---\n{page_text}")
        return "\n".join(text)
    except Exception as e:
        return f"[ERRO NO PDF: {str(e)}]"

def extract_text_from_html(file_path: Path) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
            for script in soup(["script", "style"]):
                script.decompose()
            return soup.get_text(separator='\n')
    except Exception as e:
        return f"[ERRO NO HTML: {str(e)}]"

# --- FERRAMENTAS MCP (TOOLS) ---

@mcp.tool()
def list_available_files() -> List[str]:
    """Lista arquivos na pasta de processos."""
    try:
        files = [f.name for f in TARGET_DIRECTORY.iterdir() if f.is_file()]
        return sorted(files) if files else ["Nenhum arquivo encontrado."]
    except Exception as e:
        return [f"Erro: {str(e)}"]

@mcp.tool()
def read_file_content(filename: str) -> str:
    """Lê conteúdo de PDF, HTML ou TXT."""
    try:
        file_path = (TARGET_DIRECTORY / filename).resolve()
        if not file_path.is_relative_to(TARGET_DIRECTORY.resolve()):
            return "ERRO DE SEGURANÇA."
        
        if not file_path.exists():
            return "Arquivo não encontrado."

        suffix = file_path.suffix.lower()
        if suffix == '.pdf':
            return extract_text_from_pdf(file_path)
        elif suffix == '.html':
            return extract_text_from_html(file_path)
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    except Exception as e:
        return f"Erro: {str(e)}"

# --- NOVA FERRAMENTA DE ESCRITA ---
@mcp.tool()
def save_document(filename: str, content: str) -> str:
    """
    Salva um novo documento na pasta de processos.
    Use esta ferramenta para escrever relatórios, minutas ou resumos.
    O formato ideal é Markdown (.md) para manter formatação.
    """
    try:
        # Segurança: Garante que só salva na sandbox
        safe_name = os.path.basename(filename)
        file_path = TARGET_DIRECTORY / safe_name
        
        print(f"--- 💾 SALVANDO ARQUIVO: {safe_name} ---")
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
            
        return f"Sucesso: Arquivo '{safe_name}' salvo com sucesso em {TARGET_DIRECTORY}."
    except Exception as e:
        return f"Erro ao salvar: {str(e)}"

if __name__ == "__main__":
    mcp.run()