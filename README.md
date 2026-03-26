# AI for SDGs Compliance Assessment
## Background
Setiap tahun Times Higher Education (THE) selalu mengadakan SDGs Impact Ratings kepada seluruh Universitas di seluruh negara yang terdaftar keanggotaan PBB. Penilaian (ratings) dilakukan berdasarkan bukti-bukti dokumen yang valid dan diunggah secara publik oleh universitas. Untuk masuk dalam peringkat keseluruhan, universitas wajib mengisi minimal 4 SDG. UNESA memiliki 4 SDG yang menjadi fokus utama yaitu SDG 4, 5, 8, dan 17.

## Tujuan
Mempermudah pengumpulan dokumen-dokumen administrasi UNESA sebagai bukti dalam proses scoring SDGs Impact Ratings oleh Times Higher Education (THE). Fungsi AI Agent untuk mengelompokkan dan memberikan skor kepada dokumen administrasi secara otomatis dengan hasil yang akurat dan konsisten, serta memberikan kritik atau saran yang sesuai.

## Permasalahan Teknis
1. AI harus tau apa saja yang menjadi indikator penilaian oleh THE karena halusinasi pada AI yang menyebabkan pengelompokkan dan penilaian pada dokumen menjadi tidak konsisten dan tidak sesuai dengan indikator penilaian.
2. Beberapa dokumen administrasi memiliki gambar, tabel, dan struktur penulisan yang beragam yang menuntut proses ekstraksi dokumen harus benar-benar merepresentasikan isi dokumen.
3. Keterbatasan resource terutama pada penggunaan AI seperti jumlah token dan jumlah request yang mana perlu adanya optimasi sistem.


## Model
List beberapa model yang nantinya akan digunakan
### Sentence Embedding Model
1. `QWEN3-Embedding-0.6B` : Masuk dalam peringkat 9 di MTEB multilingual leaderboard huggingface dan merupakan model yang sangat ringan.
2. `gemini-embedding-001` : Free API sebesar 1000 request per hari dan token limit sebesar 30k per menit.
3. `jina-embeddings-v5-text-nano` : Masuk dalam peringkat 10 di MTEB multilingual leaderboard huggingface dengan parameter hanya 0.22B
### LLM (Agent)
1. QWEN3-Coder-Next
2. Gemini-2.5-flash
3. Deepseek