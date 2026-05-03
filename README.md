# Analisis Pengaruh Panjang Chunk dan Overlap terhadap Akurasi Jawaban Small Language Model (SLM) pada Dokumen Publik Berbahasa Indonesia

**Penulis:** Fikri  
**NPM:** 2024171004  
**Program Studi:** Magister Informatika, Universitas Indo Global Mandiri Palembang  
**Pembimbing:** Rudi Heriansyah, S.T., M.Eng., Ph.D

---

## 📋 Ringkasan Penelitian

Penelitian ini menganalisis pengaruh strategi **chunking** (pemotongan dokumen) terhadap akurasi jawaban **Small Language Models (SLM)** pada sistem **Retrieval-Augmented Generation (RAG)** untuk dokumen publik berbahasa Indonesia. Fokus utama adalah menemukan kombinasi optimal antara **panjang chunk** (128, 256, 512 token) dan **persentase overlap** (0%, 10%, 20%) yang menghasilkan performa terbaik.

### Mengapa Penelitian Ini Penting?

- **Isu Biaya & Infrastruktur:** LLM besar (GPT-4, dll.) membutuhkan biaya API tinggi dan infrastruktur cloud — tidak semua organisasi memilikinya.
- **Privasi Data:** Banyak perusahaan tidak ingin data sensitif dikirim ke API pihak ketiga.
- **Low-Resource Language:** Bahasa Indonesia masih *underrepresented* dalam riset NLP global; temuan untuk bahasa Inggris belum tentu berlaku langsung.
- **Solusi Praktis:** SLM dapat dijalankan secara lokal di laptop standar, dan RAG mengatasi keterbatasan pengetahuan model.

---

## 🎯 Tujuan Penelitian

1. Menganalisis pengaruh signifikan panjang chunk dan overlap terhadap metrik evaluasi menggunakan kerangka kerja **RAGAS**.
2. Menentukan konfigurasi parameter chunking optimal untuk dokumen berbahasa Indonesia.
3. Membandingkan efektivitas tiga model SLM dalam tugas Question Answering berbasis RAG pada lingkungan komputasi terbatas (laptop).

---

## 🔬 Variabel Penelitian

| Variabel | Nilai yang Diuji |
|----------|------------------|
| **Chunk Size** | 128 token, 256 token, 512 token |
| **Overlap** | 0%, 10%, 20% |
| **SLM Models** | Qwen2.5-3B-Instruct, Sailor2-3B-Chat, SEA-LION-v1-3B |
| **Judge Model (RAGAS)** | Kimi K2.5 (via API) |

---

## 📁 Struktur Repositori

```
my_thesis/
├── main.tex                          # File utama LaTeX
├── referensi.bib                     # Daftar pustaka (BibTeX)
├── README.md                         # Dokumen ini
├── CHANGELOG.md                      # Riwayat perubahan
│
├── chapters/
│   ├── 01 - pendahuluan.tex          # BAB I: Latar Belakang, Rumusan Masalah, Tujuan
│   ├── 02 - tinjauan pustaka.tex     # BAB II: Landasan Teori (LLM, SLM, RAG, Chunking, Evaluasi)
│   └── 03 - metodologi penelitian.tex # BAB III: Metode, Desain Eksperimen, Metrik, Prosedur
│
└── [chapter lain akan ditambahkan]
```

---

## 🛠️ Spesifikasi Teknis

### Lingkungan Pengembangan

| Komponen | Spesifikasi |
|----------|-------------|
| **CPU** | Intel Core i7 Evo |
| **RAM** | 16 GB |
| **GPU** | Intel Iris Xe (Onboard) |
| **OS** | Linux / Windows |

### Tools & Framework

- **Pipeline RAG:** LangChain / LlamaIndex
- **Inference Engine:** llama.cpp / Ollama
- **Model Format:** GGUF (4-bit / 8-bit quantization)
- **Vector Database:** FAISS
- **Embedding Model:** `intfloat/multilingual-e5-large`
- **Evaluation Framework:** RAGAS (LLM-as-a-Judge)
- **Judge Model:** Kimi K2.5 via API

---

## 📊 Dataset

| Sumber | Domain | Deskripsi |
|--------|--------|-----------|
| **UU ITE** | Hukum | Undang-Undang Nomor 11 Tahun 2008 dan UU Nomor 1 Tahun 2024 tentang Informasi dan Transaksi Elektronik. Dokumen formal, padat, kompleks secara struktural. |
| **Indonesian News Corpus** | Berita | Kumpulan artikel berita berbahasa Indonesia dari berbagai portal berita. Teks jurnalistik dengan variasi topik dan gaya penulisan. |

---

## 📈 Metrik Evaluasi

### Metrik Retrieval
- **Recall@k:** Persentase pertanyaan dengan minimal satu chunk relevan dalam top-k
- **Precision@k:** Proporsi chunk relevan di antara top-k hasil

### Metrik End-to-End RAG (RAGAS)
- **Faithfulness:** Sejauh mana jawaban didukung oleh fakta dalam konteks (anti-halusinasi)
- **Answer Relevancy:** Sejauh mana jawaban langsung menjawab pertanyaan
- **Context Precision:** Rasio sinyal-terhadap-noise pada chunk yang diambil
- **Context Recall:** Kemampuan sistem menemukan seluruh informasi yang diperlukan

### Metrik Efisiensi
- **Average Latency:** Waktu dari input pertanyaan hingga jawaban akhir (detik)
- **Peak Memory Usage:** Penggunaan memori maksimum selama inferensi

---

## 📅 Timeline Penelitian

| Periode | Kegiatan |
|---------|----------|
| **Januari 2026** | Pengumpulan dokumen, cleaning, pembuatan Ground Truth QA |
| **Januari–Februari 2026** | Pengembangan sistem RAG dan instalasi environment |
| **Februari 2026** | Pelaksanaan eksperimen (9 skenario × 3 model) |
| **Februari–Maret 2026** | Evaluasi metrik, analisis data, kesimpulan |

---

## 🔗 Referensi Utama

- Lewis et al. (2020) — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*
- Es et al. (2025) — *RAGAS: Automated Evaluation of Retrieval Augmented Generation*
- Zheng et al. (2023) — *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*
- Zhu et al. (2026) — *Best Practices for RAG*
- Cahyawijaya et al. (2021) — *IndoNLG: Benchmark & Resources for Indonesian NLG*

---

## 📝 Catatan Pengembangan

- Thesis ditulis dalam format **LaTeX** menggunakan paket `babel` untuk dukungan bahasa Indonesia.
- Font utama: **Times New Roman** (12pt).
- Spasi: **1.5 baris** (untuk proposal).
- Margin: Kiri 4cm, lainnya 3cm.

---

## ⚠️ Batasan Penelitian

- Fokus pada dokumen berbahasa Indonesia.
- Tiga model SLM dengan ukuran ~3B parameter.
- Lingkungan komputasi terbatas (laptop standar, tidak ada GPU diskrit).

---

## 📧 Kontak

- **LinkedIn / Professional:** [PT. Qiscus Tekno Indonesia](https://www.qiscus.com)
- **Repository GitHub:** [asahfikir/my_thesis](https://github.com/asahfikir/my_thesis)

---

*Proposal Tesis — Program Magister Informatika, Universitas Indo Global Mandiri Palembang, 2026.*
