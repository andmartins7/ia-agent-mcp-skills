import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Configuração dos parâmetros do servidor
server_params = StdioServerParameters(
    command="uv", # Usamos UV para garantir o ambiente
    args=["run", "file_server.py"], # O comando que roda seu servidor
    env=None
)

async def run_test():
    print("--- 🔌 Conectando ao Servidor MCP via STDIO ---")
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 1. Handshake (Inicialização)
            await session.initialize()
            
            # 2. Listar Ferramentas Disponíveis
            print("\n--- 🛠️  Listando Ferramentas ---")
            tools = await session.list_tools()
            for tool in tools.tools:
                print(f"- {tool.name}: {tool.description}")

            # 3. Testar a Ferramenta 'list_available_files'
            print("\n--- 📂 Executando: list_available_files ---")
            result_list = await session.call_tool("list_available_files", arguments={})
            print(f"Resultado: {result_list.content[0].text}")

            # 4. Testar a Ferramenta 'read_file_content' (Se houver arquivos)
            # Vamos criar um arquivo dummy caso não exista para testar
            if "Nenhum arquivo" in result_list.content[0].text:
                print("\n(Criando arquivo de teste simulado...)")
                with open("dados_processos/teste.txt", "w") as f:
                    f.write("Conteúdo confidencial do processo 123.")
                
                print("--- 📖 Executando: read_file_content (teste.txt) ---")
                result_read = await session.call_tool("read_file_content", arguments={"filename": "teste.txt"})
                print(f"Conteúdo: {result_read.content[0].text}")

if __name__ == "__main__":
    # Cria a pasta necessária se não existir
    import os
    os.makedirs("dados_processos", exist_ok=True)
    
    asyncio.run(run_test())