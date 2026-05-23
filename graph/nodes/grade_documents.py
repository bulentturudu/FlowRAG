"""
================================================================================
PROJECT: FlowRAG (Advanced Graph-Based RAG System)
FILE: graph/nodes/grade_documents.py
AUTHOR: Bulent Turudu
DESCRIPTION:
    This node functions as the Document Verification Stage. It iterates through
    all retrieved documents from the vectorstore, evaluates their semantic
    relevance against the user query using the structured retrieval grader,
    filters out noise, and triggers the web search flag if any document is rejected.

graph.py'deki Akış Yeri:
    RETRIEVE ──► GRADE_DOCUMENT ──► (WEBSEARCH veya GENERATE)
================================================================================
"""

from typing import Any, Dict
from graph.chains.retrieval_grader import retrieval_grader
from graph.state import GraphState


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Vektör deposundan getirilen belgelerin soruyla alakasını değerlendirir.
    Alakasız belgeleri eler. Eğer tek bir belge bile elenirse, bilgi kaybını
    telafi etmek amacıyla 'web_search' bayrağını True yapar.
    """
    print("---- CHECK DOCUMENTS RELEVANT TO QUESTION ---- ")

    question = state["question"]
    documents = state["documents"]

    # Alaka süzgecinden başarıyla geçen temiz dökümanların toplanacağı yeni liste
    filtered_documents = []

    # Varsayılan olarak internet aramasını kapalı (False) kabul ediyoruz
    web_search = False

    # ------------------------------------------------------------------------------
    # DÖKÜMAN DEĞERLENDİRME DÖNGÜSÜ
    # ------------------------------------------------------------------------------
    for d in documents:
        # Her bir döküman parçasının metin içeriğini (`page_content`) LLM zincirine soruyoruz
        score = retrieval_grader.invoke(
            {
                "question": question,
                "document": d.page_content
            }
        )

        # 🛠️ TİP GÜVENLİĞİ VE SADELEŞTİRME GÜNCELLEMESİ
        # KENDİME NOT: retrieval_grader.py dosyasında çıktı şemasını string yerine
        # doğrudan 'bool' yaptığımız için, eski koddaki `score.binary_score.lower() == "yes"`
        # gibi kırılgan string karşılaştırmalarını çöpe attık. Doğrudan True/False okuyoruz.
        grade = score.binary_score

        if grade:
            print("----GRADE: DOCUMENT RELEVANT----")
            # Eğer belge soruyla alakalıysa (True), filtrelenmiş güvenli listeye ekle
            filtered_documents.append(d)
        else:
            print("----GRADE: DOCUMENT NOT RELEVANT----")
            # KENDİME NOT: Alakasız bir döküman yakalandığı an bu dökümanı eliyoruz
            # (filtered_documents listesine EKLEMİYORUZ). Ayrıca eksik kalan bilgiyi
            # internetten (Tavily) tamamlamak için web_search bayrağını True yapıyoruz.
            web_search = True

    # ------------------------------------------------------------------------------
    # STATE GÜNCELLEMESİ VE GERİ DÖNÜŞ
    # ------------------------------------------------------------------------------
    # Temizlenmiş döküman listesini ve internet arama kararını (web_search)
    # grafiğin ortak hafızasına (State) geri yazıyoruz.
    return {
        "question": question,
        "documents": filtered_documents,
        "web_search": web_search,
    }