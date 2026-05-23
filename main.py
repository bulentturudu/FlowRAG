"""
================================================================================
PROJECT: FlowRAG (Advanced Graph-Based RAG System)
FILE: main.py
AUTHOR: Bulent Turudu
DESCRIPTION:
    This file serves as the main execution gateway for the FlowRAG application.
    It initializes the environment variables, triggers the compiled LangGraph
    state machine with a user query, and manages the orchestration flow.

KENDİME NOT:
    Projenin ana tetikleyicisi burasıdır. Grafiğin (Graph) başlatılması,
    farklı node'ların (retrieve, web_search, generate) bir sıra halinde
    koşması ve en nihayetinde state'in son halinin ekrana basılması bu dosya
    üzerinden yönetilir.
================================================================================
"""

from dotenv import load_dotenv
from graph.graph import app
from rich.console import Console

# ------------------------------------------------------------------------------
# 1. ORTAM DEĞİŞKENLERİ VE YAPILANDIRMA
# ------------------------------------------------------------------------------
# .env dosyasındaki OpenAI, Tavily vb. API anahtarlarını sisteme yükler.
# Projedeki tüm LLM ve Search Tool yapıları bu anahtarları arkada otomatik okur.
load_dotenv()

# Console
console = Console(force_terminal=True)

# ------------------------------------------------------------------------------
# 2. ANA UYGULAMA TETİKLEYİCİSİ (MAIN ENTRYPOINT)
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    # KENDİME NOT: Test etmek istediğin soruyu aşağıdaki 'question' alanına yazabilirsin.
    # Router (Yönlendirici) mekanizması sorunun tipine göre (RAG mı, Genel Kültür mü)
    # otomatik karar verip doğru düğüme (node) zıplayacaktır.
    test_query = {
        "question": "What is the difference between Chain of Thought (CoT) and Tree of Thoughts (ToT) in the context of an agent's planning capability, and how does the ReAct framework integrate planning with action?"
        # "question": "how can i make hamburger?"  # Doğrudan Web Search Testi
    }

    console.print("\n--- FlowRAG Execution Started ---\n", style="bold blue")

    # Derlenmiş LangGraph iş akışını (StateGraph App) girdiyle tetikliyoruz.
    # invoke() işlemi bittiğinde tüm düğümlerden geçen 'state' sözlüğü geri döner.
    final_state = app.invoke(input=test_query)

    #console.print("\n--- Final Answer ---", style="bold green")
    #console.print(final_state.get("generation", "No answer found."), style="yellow")

    console.print("\n--- Full State (Debug) ---", style="bold blue")
    console.print(final_state, style="dim")
