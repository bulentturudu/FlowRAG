"""
================================================================================
PROJECT: FlowRAG (Advanced Graph-Based RAG System)
FILE: ingestion.py
AUTHOR: Bulent Turudu
DESCRIPTION:
    This module performs ETL (Extract, Transform, Load) operations for FlowRAG.
    It scrapes target URLs, aggressively deep-cleans HTML markup using BeautifulSoup
    to remove noise (navbars, footers, scripts, etc.), splits pure text using
    a token-aware recursive text splitter, and indexes chunks into ChromaDB.

KENDİME NOT:
    Eğer yerel veritabanına yeni makaleler veya dokümanlar eklemek istersen,
    tek yapman gereken 'urls' listesine yeni linkler ekleyip bu dosyayı
    `python ingestion.py` diyerek çalıştırmaktır. Klasör çakışması veya bayat
    veri kalıntısı olmaması için öncesinde terminalden `rm -rf .chroma` ile
    eski klasörü silmeyi unutma!
================================================================================
"""

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# .env dosyasındaki OpenAI API anahtarını yükler (Embedding hesaplaması için zorunlu)
load_dotenv()

# ------------------------------------------------------------------------------
# KONTROL PANELİ: Hedef Bilgi Kaynakları
# ------------------------------------------------------------------------------
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-1Lm/"
]


def run_ingestion():
    print("---- INGESTION STARTED: Cleaning HTML with BeautifulSoup ----")

    all_splits = []

    # ------------------------------------------------------------------------------
    # METİN PARÇALAMA STRATEJİSİ (TEXT SPLITTER)
    # ------------------------------------------------------------------------------
    # KENDİME NOT: Tiktoken tabanlı bölücü karakter sayısına göre değil, LLM'in
    # okuduğu token yapısına göre akıllıca böler.
    # chunk_size=500: Bağlam bütünlüğü kaybolmasın diye ideal bir büyüklüktür.
    # chunk_overlap=50: Parçalar arası cümle geçişlerinde bilgi kaybını önler.
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500,
        chunk_overlap=50
    )

    for url in urls:
        print(f"Fetching and deep-cleaning content: {url}")
        response = requests.get(url)

        # Ham HTML içeriğini parse et
        soup = BeautifulSoup(response.text, "html.parser")

        # ------------------------------------------------------------------------------
        # 🛠️ GÜVENLİK FİLTRESİ: Gürültü ve Çöp Temizliği
        # ------------------------------------------------------------------------------
        # KENDİME NOT: LLM'i sinsi yönlendirme butonları, alt bilgi linkleri veya
        # menü kelimeleriyle ("nav", "footer", "aside") zehirlememek için bu HTML
        # etiketlerini .extract() ile sayfadan tamamen söküp fırlatıyoruz.
        for element in soup(["nav", "footer", "header", "script", "style", "aside"]):
            element.extract()

        # Lilian Weng'in blog mimarisinde ana yazı gövdesi genellikle <article>
        # veya class="post-content" içindedir. Nokta atışı burayı çekiyoruz.
        main_content = soup.find("article") or soup.find(class_="post-content")

        if main_content:
            clean_text = main_content.get_text(separator="\n")
        else:
            # Eğer özel bir gövde kapsayıcısı bulunamazsa tüm temiz metni yedek olarak al
            clean_text = soup.get_text(separator="\n")

        # Temizlenmiş pürüzsüz metni LangChain'in anlayacağı Document formatına sarıyoruz
        doc = Document(
            page_content=clean_text,
            metadata={"source": url}
        )

        # Dokümanı yukarıda tanımladığımız token kurallarına göre küçük parçalara (splits) bölüyoruz
        url_splits = text_splitter.split_documents([doc])
        all_splits.extend(url_splits)

    print(f"---- SAVING TO DATABASE: Writing {len(all_splits)} pure-content chunks to ./.chroma ----")

    # ------------------------------------------------------------------------------
    # VEKTÖR VERİTABANINA YAZMA (CHROMA INDEXING)
    # ------------------------------------------------------------------------------
    # KENDİME NOT: En güncel `langchain_chroma` paketini kullanıyoruz. Temizlenmiş
    # 47 parçanın her birini OpenAIEmbeddings() ile vektör matrisine çevirip diskteki
    # .chroma klasörüne kalıcı (persist) olarak indeksliyoruz.
    vectorstore = Chroma.from_documents(
        documents=all_splits,
        collection_name="rag-chroma",
        embedding=OpenAIEmbeddings(),
        persist_directory="./.chroma"
    )

    print("---- INGESTION COMPLETED SUCCESSFULLY ----")
    return vectorstore


if __name__ == '__main__':
    run_ingestion()