# FlowRAG

LangGraph tabanlı, akıllı yönlendirme yapan bir RAG (Retrieval-Augmented Generation) sistemi.

Gelen soruyu analiz ederek ya yerel vektör veritabanından belge getirir ya da web araması yapar. Getirilen belgeler alakalılık açısından puanlanır, üretilen cevap halüsinasyon kontrolünden geçirilir.

---

## Nasıl Çalışır

```
Soru → Router → [RAG veya Web Search] → Document Grading → Generate → Hallucination Check → Cevap
```

---

## Kurulum

### 1. Repoyu klonla

```bash
git clone https://github.com/kullanici-adi/FlowRAG.git
cd FlowRAG
```

### 2. Sanal ortam oluştur

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 3. Bağımlılıkları yükle

```bash
# Geliştirme ortamı için
pip install -r requirements/dev.txt

# Production için
pip install -r requirements/prod.txt

# Ya da direkt
pip install -r requirements.txt
```

### 4. Ortam değişkenlerini ayarla

`.env.example` dosyasını kopyala ve API key'lerini ekle:

```bash
cp .env.example .env
```

```env
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

---

## Kullanım

### Belge yükleme (Ingestion)

```bash
python ingestion.py
```

### Sorgu çalıştırma

`main.py` içindeki `test_query` alanını düzenle:

```python
test_query = {
    "question": "Sorunuzu buraya yazın."
}
```

Sonra çalıştır:

```bash
python main.py
```

---

## Proje Yapısı

```
FlowRAG/
├── graph/
│   ├── chains/        # LLM zincirleri (grader, router, generation)
│   ├── nodes/         # Graf düğümleri (retrieve, generate, web_search...)
│   ├── graph.py       # LangGraph iş akışı
│   ├── state.py       # Graf state tanımı
│   └── config.py      # Ayarlar
├── requirements/
│   ├── base.txt
│   ├── dev.txt
│   └── prod.txt
├── ingestion.py       # Belge yükleme
├── main.py            # Giriş noktası
└── requirements.txt
```

---

## Gereksinimler

- Python 3.11+
- OpenAI API key
- Tavily API key
