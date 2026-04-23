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
List beberapa model yang digunakan
### Sentence Embedding Model
`microsoft/harrier-oss-v1-0.6b` : Model embedding terbaru dari micorsoft dijalankan dengan API huggingface inference dengan free tier limit yang cukup besar. Saat ini (18 April 2026) Peringkat 10 dalam MTEB leaderboard.
### LLM (Agent)
`nvidia/nemotron-3-super-120b-a12b:free` : Model LLM gratis yang disediakan oleh nvidia. API menggunakan openrouter API.

# Flowchart System
![Flowchart System](https://res.cloudinary.com/dxkxrbteg/image/upload/v1776511868/Project_Flow_tnb3a1.png)