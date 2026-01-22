import asyncio
import os
import sys
from typing import Annotated
from typing_extensions import TypedDict

# --- 1. CARREGAMENTO DE AMBIENTE ---
from dotenv import load_dotenv
load_dotenv()

# --- IMPORTAÇÕES DE IA ---
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

# Importações do MCP
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# --- 2. FÁBRICA DE MODELOS ---
def get_llm():
    temperature = float(os.getenv("MODEL_TEMPERATURE", 0))
    if os.getenv("GOOGLE_API_KEY"):
        print("--- 🧠 Cérebro: Usando Google Gemini 2.0 Flash ---")
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            temperature=temperature,
            max_retries=2,
            safety_settings={
                "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE", 
                "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE", 
                "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE", 
                "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            }
        )
    elif os.getenv("GROQ_API_KEY"):
        print("--- ⚡ Cérebro: Usando Groq (Llama 3.3 70B) ---")
        from langchain_groq import ChatGroq
        return ChatGroq(model="llama-3.3-70b-versatile", temperature=temperature)
    else:
        print("--- 🦙 Cérebro: Usando Ollama Local ---")
        from langchain_ollama import ChatOllama
        return ChatOllama(model="llama3.1", temperature=temperature)

llm = get_llm()

# --- 3. CONFIGURAÇÃO MCP ---
server_params = StdioServerParameters(
    command="uv",
    args=["run", "file_server.py"],
    env=os.environ.copy(),
)

# --- 4. O AGENTE (LÓGICA) ---
async def run_agent_session():
    # Caminho do Banco de Dados (O arquivo será criado na pasta raiz)
    DB_PATH = "memoria_tribunal.sqlite"
    
    print(f"--- 💾 Conectando à Memória Durável: {DB_PATH} ---")

    # AQUI ESTÁ A MÁGICA: Context Manager do SQLite
    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        
        # Conecta ao MCP (Mãos)
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # --- TOOLS ---
                @tool
                async def list_files_tool():
                    """Lista os arquivos de processos na pasta."""
                    result = await session.call_tool("list_available_files", arguments={})
                    return result.content[0].text

                @tool
                async def read_file_tool(filename: str):
                    """Lê o conteúdo de um arquivo. Exige o nome exato."""
                    result = await session.call_tool("read_file_content", arguments={"filename": filename})
                    return result.content[0].text

                @tool
                async def save_file_tool(filename: str, content: str):
                    """Salva um documento/relatório no disco."""
                    result = await session.call_tool("save_document", arguments={"filename": filename, "content": content})
                    return result.content[0].text

                tools = [list_files_tool, read_file_tool, save_file_tool]
                llm_with_tools = llm.bind_tools(tools)

                # --- GRAFO ---
                class State(TypedDict):
                    messages: Annotated[list, add_messages]

                async def agent_node(state: State):
                    return {"messages": [await llm_with_tools.ainvoke(state["messages"])]}

                async def tools_node(state: State):
                    last_message = state["messages"][-1]
                    outputs = []
                    for tool_call in last_message.tool_calls:
                        print(f"   🔨 Tool Call: {tool_call['name']}")
                        if tool_call["name"] == "list_files_tool":
                            res = await list_files_tool.ainvoke(tool_call["args"])
                        elif tool_call["name"] == "read_file_tool":
                            res = await read_file_tool.ainvoke(tool_call["args"])
                        elif tool_call["name"] == "save_file_tool":
                            res = await save_file_tool.ainvoke(tool_call["args"])
                        else: res = "Erro."
                        outputs.append(ToolMessage(content=str(res), tool_call_id=tool_call["id"], name=tool_call["name"]))
                    return {"messages": outputs}

                def should_continue(state: State):
                    return "tools" if state["messages"][-1].tool_calls else END

                workflow = StateGraph(State)
                workflow.add_node("agent", agent_node)
                workflow.add_node("tools", tools_node)
                workflow.add_edge(START, "agent")
                workflow.add_conditional_edges("agent", should_continue)
                workflow.add_edge("tools", "agent")
                
                # COMPILAÇÃO: Aqui injetamos o SQLite Checkpointer
                app = workflow.compile(checkpointer=checkpointer)

                # --- LOOP INTERATIVO ---
                print("\n--- ⚖️  AGENTE JURÍDICO (Memória Persistente Ativa) ---")
                print(f"--- Thread ID: 'juiz_principal' (Sempre retomará daqui) ---")
                
                # Configuração da Thread (Fixa para teste de persistência)
                config = {"configurable": {"thread_id": "juiz_principal"}}

                system_instruction = """
                Você é um Juiz Assistente Sênior. 
                1. Use 'read_file_tool' para ler provas.
                2. Use 'save_file_tool' APENAS uma vez para gerar minutas finais.
                3. Se o usuário perguntar sobre algo que já discutimos, USE SUA MEMÓRIA.
                """

                while True:
                    try:
                        user_input = input("\n👤 Juiz: ")
                        if user_input.lower() in ["sair", "exit"]:
                            print("Salvando estado e encerrando...")
                            break
                        
                        input_msg = {"messages": [
                            SystemMessage(content=system_instruction),
                            HumanMessage(content=user_input)
                        ]}

                        async for event in app.astream(input_msg, config=config):
                            # Debug visual simples
                            pass 
                        
                        # Pega o estado final da memória (Snapshot)
                        snapshot = await app.aget_state(config)
                        last_msg = snapshot.values["messages"][-1]
                        print(f"\n🤖 Agente: {last_msg.content}")

                    except Exception as e:
                        print(f"Erro no loop: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(run_agent_session())
    except KeyboardInterrupt:
        print("\nEncerrado.")