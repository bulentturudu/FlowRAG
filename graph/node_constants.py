"""
================================================================================
PROJECT: FlowRAG (Advanced Graph-Based RAG System)
FILE: graph/node_constants.py
AUTHOR: Bulent Turudu
DESCRIPTION:
    Centralized configuration file for LangGraph node and edge identifiers.
    Defines structural constants to avoid hardcoded string errors across the project.

KENDİME NOT:
    Grafik yapısında düğüm isimlerini ('retrieve', 'generate' vb.) el yazısıyla
    her yere string olarak dağıtmak büyük projelerde bug'lara yol açar (örn: 'web_search'
    yerine yanlışlıkla 'websearch' yazmak sistemi çökertir). Bu dosya sayesinde
    isimleri tek bir merkezi sabit havuzunda topluyoruz.
================================================================================
"""

# KENDİME NOT: ChromaDB vektör deposundan soruyla ilgili belgeleri çeken düğümün adı.
RETRIEVE = "retrieve"

# KENDİME NOT: Gelen belgelerin soruya gerçekten cevap verip vermediğini LLM ile
# puanlayan/filtreleyen (Document Grader) düğümün adı.
GRADE_DOCUMENT = "grade_document"

# KENDİME NOT: Elindeki doğrulanmış belgeleri (veya internet özetlerini) kullanarak
# LLM ile kullanıcının sorusuna nihai cevabı hazırlayan düğümün adı.
GENERATE = "generate"

# KENDİME NOT: Yerel belgeler yetersiz kaldığında Tavily Search motorunu tetikleyip
# internetten taze bilgi toplayan düğümün adı.
WEBSEARCH = "websearch"