import asyncio
import os
import sys
from typing import Annotated
from typing_extensions import TypedDict

# --- 1. CARREGAMENTO DE AMBIENTE (NOVO) ---
from dotenv import load_dotenv

# Carrega as variáveis do arquivo .env para o os.environ
load_dotenv()

# --- IMPORTAÇÕES DE IA ---
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

# Importações do MCP
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# --- 2. FÁBRICA DE MODELOS (MODEL FACTORY) ---
def get_llm():
    """
    Decide qual LLM usar baseado nas chaves disponíveis no .env.
    """
    temperature = float(os.getenv("MODEL_TEMPERATURE", 0))

    # Opção A: Google Gemini
    if os.getenv("GOOGLE_API_KEY"):
        print("--- 🧠 Cérebro: Usando Google Gemini 2.0 Flash ---")
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        return ChatGoogleGenerativeAI(
            # ATUALIZADO: Usando o modelo confirmado na lista
            model="gemini-2.0-flash", 
            temperature=temperature,
            max_retries=2,
            # Configuração de segurança para evitar bloqueio indevido em textos jurídicos (crimes, danos, etc)
            safety_settings={
                "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE", 
                "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE", 
                "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE", 
                "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            }
        )
    
    # Opção B: Groq (Atualizado para Llama 3.3)
    elif os.getenv("GROQ_API_KEY"):
        print("--- ⚡ Cérebro: Usando Groq (Llama 3.3 70B) ---")
        from langchain_groq import ChatGroq
        return ChatGroq(
            # ID do modelo
            model="llama-3.3-70b-versatile", 
            temperature=temperature
        )
    
    # Opção C: Local (Fallback)
    else:
        print("--- 🦙 Cérebro: Usando Ollama Local ---")
        from langchain_ollama import ChatOllama
        return ChatOllama(model="llama3.1", temperature=temperature)

# Inicializa o LLM
llm = get_llm()

# Inicializa a memória
memory = MemorySaver()

# --- 3. CONFIGURAÇÃO MCP (AS MÃOS) ---
server_params = StdioServerParameters(
    command="uv",
    args=["run", "file_server.py"],
    env=os.environ.copy(), # Passa as credenciais adiante se necessário
)

# --- 4. O AGENTE (LÓGICA) ---
async def run_agent_process(user_query: str):
    print(f"--- 🚀 Iniciando Processo: '{user_query}' ---")
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # --- DEFINIÇÃO DAS TOOLS ---
            @tool
            async def list_files_tool():
                """Lista os arquivos de processos na pasta."""
                # O servidor MCP faz o trabalho pesado
                result = await session.call_tool("list_available_files", arguments={})
                return result.content[0].text

            @tool
            async def read_file_tool(filename: str):
                """Lê o conteúdo de um arquivo. Exige o nome exato (ex: 'caso_123.txt')."""
                result = await session.call_tool("read_file_content", arguments={"filename": filename})
                return result.content[0].text
            
            @tool
            async def save_file_tool(filename: str, content: str):
                """Salva um arquivo no disco. Use para gerar o relatório final."""
                result = await session.call_tool("save_document", arguments={"filename": filename, "content": content})
                return result.content[0].text

            # Binding
            tools = [list_files_tool, read_file_tool, save_file_tool]
            llm_with_tools = llm.bind_tools(tools)

            # --- DEFINIÇÃO DO GRAFO (LANGGRAPH) ---
            class State(TypedDict):
                messages: Annotated[list, add_messages]

            async def agent_node(state: State):
                return {"messages": [await llm_with_tools.ainvoke(state["messages"])]}

            async def tools_node(state: State):
                last_message = state["messages"][-1]
                outputs = []
                for tool_call in last_message.tool_calls:
                    print(f"   🔨 Executando Tool: {tool_call['name']}")
                    
                    if tool_call["name"] == "list_files_tool":
                        res = await list_files_tool.ainvoke(tool_call["args"])
                    elif tool_call["name"] == "read_file_tool":
                        res = await read_file_tool.ainvoke(tool_call["args"])
                    elif tool_call["name"] == "save_file_tool":
                        res = await save_file_tool.ainvoke(tool_call["args"])
                    else:
                        res = "Erro: Ferramenta desconhecida."
                    
                    outputs.append(ToolMessage(
                        content=str(res), 
                        tool_call_id=tool_call["id"], 
                        name=tool_call["name"]
                    ))
                return {"messages": outputs}

            def should_continue(state: State):
                return "tools" if state["messages"][-1].tool_calls else END

            workflow = StateGraph(State)
            workflow.add_node("agent", agent_node)
            workflow.add_node("tools", tools_node)
            
            workflow.add_edge(START, "agent")
            workflow.add_conditional_edges("agent", should_continue)
            workflow.add_edge("tools", "agent")
            
            # Acoplamos a memória global ao grafo
            app = workflow.compile(checkpointer=memory)

            # --- EXECUÇÃO (COM STOP CONDITION) ---
            system_instruction = """
            Você é um Juiz Assistente Sênior. Siga este processo RÍGIDO:

            1. LEITURA: Use 'read_file_tool' para ler o conteúdo completo do arquivo solicitado.
            2. RASCUNHO MENTAL: Analise os fatos, argumentos e decisão.
            3. CRIAÇÃO: Gere o texto completo do relatório. O texto deve ter:
               - Cabeçalho
               - Relatório dos Fatos (detalhado)
               - Dispositivo da Sentença (quem ganhou e valores)
            4. SALVAMENTO: Chame a ferramenta 'save_file_tool'. 
               ATENÇÃO: O parâmetro 'content' DEVE conter o TEXTO COMPLETO gerado no passo 3. 
               NUNCA escreva apenas um resumo ou título no arquivo. O arquivo deve ser útil para o juiz assinar.
            
            Após salvar, encerre com a resposta: "Minuta [nome] gerada com [x] caracteres."
            """
            # Configuração da Thread (Identificador da Conversa)
            # Em produção, isso viria de uma variável de sessão do usuário
            config = {"configurable": {"thread_id": "juiz_principal"}}

            # O LangGraph vai carregar o histórico dessa thread automaticamente
            final_state = await app.ainvoke({
                "messages": [
                    # O SystemMessage pode ser enviado sempre, o LangGraph gerencia a duplicidade
                    SystemMessage(content=system_instruction),
                    HumanMessage(content=user_query)
                ]
            }, config=config) # <--- IMPORTANTE: Passe a config aqui!
            
            print(f"\n🤖 RESPOSTA FINAL:\n{final_state['messages'][-1].content}")

if __name__ == "__main__":
    try:
        # Verifica se o .env existe (Boas práticas de DevOps)
        if not os.path.exists(".env"):
            print("⚠️ AVISO: Arquivo .env não encontrado. Verifique suas chaves.")

        print("\n--- ⚖️  AGENTE JURÍDICO ATIVO (Modo Interativo) ---")
        print("--- Suporta: .txt, .pdf, .html ---")
        
        while True:
            # Loop infinito para você testar várias perguntas sem reiniciar
            query = input("\n👤 Digite sua ordem (ou 'sair'): ")
            
            if query.lower() in ["sair", "exit", "quit"]:
                print("Encerrando...")
                break
            
            if not query.strip():
                continue
                
            # Executa o ciclo do LangGraph para cada pergunta
            asyncio.run(run_agent_process(query))
            
    except KeyboardInterrupt:
        print("\nEncerrado pelo usuário.")