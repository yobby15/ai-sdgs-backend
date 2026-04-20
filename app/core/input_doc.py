from app.utils import text_processing

from langchain_core.documents import Document

import pymupdf4llm
import json
import logging
from typing import Literal, Union, Optional
import re
import os

logger = logging.getLogger(__name__)

def convert_to_md(path: str, page_range:list[int, int] = None) -> str:
    logger.debug("[START] : Converting Pdf to MD format")
    if page_range:
        extracted_page = [i for i in range(page_range[0], page_range[1])]
    else:
        extracted_page = None
    md_text = pymupdf4llm.to_markdown(
        path,
        header=False, 
        footer=False, 
        pages=extracted_page, 
        ignore_images=True, 
        page_separators=True
        )
    logger.debug("[END] : Converting Pdf to MD format (string length:%s)", str(len(md_text)))
    return md_text


def detect_metadata(pages_list: list[str]) -> dict:
    """
    Function untuk mendeteksi keberadaan metadata

    `page_list`: list berisi text tiap halaman

    **return** dictionary berisi beberapa index halaman untuk metadata.
    """

    logger.debug(f"[START] : Metadata Detection (total page: {len(pages_list)})")
    metadata_indices = {
        "title_page": None,
        "abstract_page": None,
        "conclusion_page": None,
        "reference_page": None,
    }

    # Cari ABSTRAK
    for idx, page_content in enumerate(pages_list):
        if re.search(text_processing.ABSTRACT_PATTERN, page_content):

            metadata_indices["abstract_page"] = idx
            logger.debug(f"Abstract found in index page = {metadata_indices["abstract_page"]}")
            
            # Logika penentuan halaman judul
            content_before_abs = re.split(r'(?i)abstract|abstrak', page_content)[0].strip()
            if len(content_before_abs) < 50 and idx > 0:
                metadata_indices["title_page"] = idx - 1
            else:
                metadata_indices["title_page"] = idx
            
            logger.debug(f"Title found in index page = {metadata_indices["title_page"]}")
            break

    # Jika abstrak tidak ditemukan, kita mulai pencarian dari halaman 0 dan halaman title juga 0
    if metadata_indices['abstract_page'] is None:
        logger.warning("Abstract not found and title index page set to 0!")

        metadata_indices['title_page'] = 0
        start_search = 0
    else:
        start_search = metadata_indices['abstract_page']
    
    # Cari CONCLUSION
    fallback_discussion = None
    for idx in range(start_search, len(pages_list)):
        page_content = pages_list[idx]
        
        if metadata_indices["conclusion_page"] is None:
            if re.search(text_processing.CONCLUSION_PATTERN, page_content):
                metadata_indices["conclusion_page"] = idx
                start_search = idx
                
                logger.debug(f"Conclusion found in index page : {metadata_indices["conclusion_page"]}")
                break

        # Sebagai fallback apabila dokumen tidak memiliki conclusion
        if re.search(text_processing.DISCUSSION_PATTERN, page_content):
            fallback_discussion = idx

    if metadata_indices["conclusion_page"] is None:
        if fallback_discussion is not None:
            logger.warning(f"Conclusion page not found and set to discussion index page = {fallback_discussion}")
            metadata_indices["conclusion_page"] = fallback_discussion
            start_search = fallback_discussion
        else:
            logger.warning(f"Conclusion and Discussion page not found!")


    # Cari Reference
    for idx in range(start_search, len(pages_list)):
        page_content = pages_list[idx]
        if metadata_indices["reference_page"] is None:
            if re.search(text_processing.REFERENCE_PATTERN, page_content):
                metadata_indices["reference_page"] = idx
                logger.debug(f"Reference found in index page : {metadata_indices["reference_page"]}")
    
    if metadata_indices["reference_page"] is None:
        logger.warning("Reference page not found!")

    logger.debug("[END] : Metadata Detection")
    return metadata_indices


def extract_metadata(text : str, doc_name : str) -> dict:
    """
    Mengekstrak metadata kunci dari dokumen akademik dan publikasi dosen.

    Fungsi ini memproses teks berformat Markdown untuk mengidentifikasi dan mengambil komponen penting seperti judul, abstrak, dan kesimpulan.

    Args:
        text (str): Konten dokumen mentah yang sudah dikonversi ke format Markdown.
        doc_name (str): Nama file atau identitas unik dokumen sebagai referensi.

    Returns:
        dict: Dictionary yang berisi metadata hasil ekstraksi dengan kunci:
            - 'document_name' (str): Nama asli dokumen.
            - 'title' (str): Judul artikel/paper yang terdeteksi.
            - 'abstract' (str): Abstrak dokumen.
            - 'conclusion' (str): Kesimpulan atau poin akhir dokumen.
    """

    splitted_pages = text_processing.split_and_clean_pages(text, add_overlap= False)
    metadata_pages = detect_metadata(splitted_pages)

    metadata = {
        "document_name": doc_name,
        "title": None,
        "abstract": None,
        "conclusion": None
    }

    # --- EKSTRAKSI JUDUL ---
    if metadata_pages['title_page'] is not None:
        text_ttl = splitted_pages[metadata_pages['title_page']]
        title = text_processing.extract_title(text_ttl)
        if title:
            metadata['title'] = text_processing.clean_markdown(title)


    # --- EKSTRAKSI ABSTRAK ---
    if metadata_pages['abstract_page'] is not None:
        text_abs = splitted_pages[metadata_pages['abstract_page']]
        abstract = text_processing.extract_abstract(text_abs)
        if abstract:
            metadata["abstract"] = text_processing.clean_markdown(abstract)
    else:
        text_abs = splitted_pages[metadata_pages['title_page']]
        abstract = text_processing.extract_abstract(text_abs)
        if abstract:
            metadata["abstract"] = text_processing.clean_markdown(abstract)
        

    # --- EKSTRAKSI KESIMPULAN ---
    if metadata_pages['conclusion_page'] is not None:
        if metadata_pages['reference_page'] is not None:
            text_con = '\n'.join(splitted_pages[metadata_pages['conclusion_page']:metadata_pages['reference_page']+1])
        else:
            text_con = '\n'.join(splitted_pages[metadata_pages['conclusion_page']:metadata_pages['conclusion_page']+1])
        conclusion = text_processing.extract_conclusion(text_con)
        metadata["conclusion"] = conclusion

    return metadata


def extract_chunk(
        text: str,
        type_doc: Literal["sdg_evidence", "sdg_knowledge"],
        source: str,
        start_page_index:int= 0,
        special_page: list[int] = [],
        page_overlap_char: int = 100,
        chunk_size: int = 500,
        chunk_overlap_char: int = 100,
        tolerance_rate: float = 0.5,
        min_chunk_size: int = 100,
) -> list[dict]:
    """
    Memproses teks mentah dokumen menjadi potongan-potongan kecil (chunks) yang terstruktur.

    Fungsi ini memiliki pipeline: 
    1. Membagi teks berdasarkan halaman dengan pembersihan karakter dan sistem overlap antar halaman.
    2. Melakukan chunking pada setiap halaman berdasarkan batas karakter yang ditentukan untuk 
       memastikan konteks tetap terjaga bagi model LLM/Embedding.

    Args:
        text (str): Konten teks mentah yang diekstrak dari dokumen.
        type_doc (Literal["sdg_evidence", "sdg_knowledge"]): Klasifikasi dokumen untuk metadata.
        source (str): Nama file atau identitas sumber dokumen.
        special_page (list[int], optional): Daftar nomor halaman yang memerlukan perlakuan 
            khusus saat ekstraksi. Defaults to [].
        page_overlap_char (int, optional): Jumlah karakter yang tumpang tindih antar 
            halaman untuk menjaga kontinuitas teks. Defaults to 100.
        chunk_size (int, optional): Target jumlah karakter maksimal dalam satu chunk. 
            Defaults to 500.
        chunk_overlap_char (int, optional): Jumlah karakter yang tumpang tindih antar 
            chunk berturutan. Defaults to 100.
        tolerance_rate (float, optional): Rasio toleransi untuk perhitungan fleksibilitas 
            ukuran chunk dan overlap (misal: 0.5 berarti toleransi 50%). Defaults to 0.5.
        min_chunk_size (int, optional): Batas minimum karakter agar sebuah chunk dianggap 
            valid dan tidak dibuang. Defaults to 100.

    Returns:
        list[dict]: Daftar objek JSON/dictionary yang berisi teks chunk beserta metadatanya.

    Example:
        >>> chunks = extract_chunk(text="isi dokumen...", type_doc="sdg_evidence", source="Laporan_2024.pdf")
        >>> print(chunks[0]['text'])
    """

    splitted_page = text_processing.split_and_clean_pages(
        text=text,
        start_page_index=start_page_index,
        special_page=special_page,
        overlap_chars=page_overlap_char,
        tolerance=round(tolerance_rate*page_overlap_char)
    )

    chunked_content_per_page = [
        text_processing.chunk_text(
            content,
            chunk_size=chunk_size,
            overlap=chunk_overlap_char,
            tolerance=round(tolerance_rate*chunk_size),
            min_chunk_size=min_chunk_size
        ) for content in splitted_page
    ]

    clean_extracted = text_processing.pages_to_json_format(
        chunked_content_per_page,
        type_doc=type_doc,
        source=source,
        start_page_index = start_page_index
    )

    return clean_extracted



def extract_document(
    path_file: str,
    type_doc: Literal["sdg_evidence", "sdg_knowledge"],
    source: str,
    save_path: Optional[str] = None,
    page_range: Optional[list[int]] = None,
    special_page: list[int] = [],
    page_overlap_char: int = 100,
    chunk_size: int = 500,
    chunk_overlap_char: int = 100,
    tolerance_rate: float = 0.5,
    min_chunk_size: int = 100,
) -> Union[dict, list[dict]]:
    """
    Orkestrator utama untuk mengekstraksi konten dokumen menjadi format terstruktur (JSON/Metadata/Chunks).

    Fungsi ini melakukan konversi dokumen ke Markdown terlebih dahulu, kemudian menerapkan logika ekstraksi yang berbeda berdasarkan kategori dokumen:
    1. **sdg_evidence**: Mencoba mengekstrak metadata (judul, abstrak, kesimpulan). Jika abstrak/kesimpulan tidak ditemukan, sistem otomatis melakukan fallback ke proses chunking.
    2. **sdg_knowledge**: Langsung melakukan proses chunking teks untuk knowledge database.

    Args:
        path_file (str): Path file sumber yang akan diproses.
        type_doc (Literal["sdg_evidence", "sdg_knowledge"]): Klasifikasi dokumen untuk menentukan alur ekstraksi.
        source (str): Nama sumber atau identitas dokumen.
        save_path (str, optional): Path lengkap (termasuk .json) jika ingin menyimpan hasil ekstraksi ke file lokal. Defaults to None.
        page_range (list[int], optional): Batasan halaman yang akan diproses, misal [awal, akhir]. Jika None, seluruh dokumen diproses. Defaults to None.
        special_page (list[int], optional): Daftar halaman tertentu yang tidak memerlukan overlap. Defaults to [].
        page_overlap_char (int, optional): Karakter overlap antar halaman. Defaults to 100.
        chunk_size (int, optional): Ukuran maksimal karakter per chunk. Defaults to 500.
        chunk_overlap_char (int, optional): Karakter overlap antar chunk. Defaults to 100.
        tolerance_rate (float, optional): Tingkat toleransi fleksibilitas ukuran teks. Defaults to 0.5.
        min_chunk_size (int, optional): Batas minimal karakter untuk validasi chunk. Defaults to 100.

    Returns:
        Union[dict, list[dict]]: Jika mengekstrak metadata, mengembalikan dictionary 
            metadata. Jika melakukan chunking, mengembalikan list of dictionaries (chunks).
    """

    logger.debug("[START] : Extracting Document (input parameter:%s)", {
        "path_file":path_file,
        "type_doc":type_doc,
        "source":source,
        "save_path":save_path,
        "page_range":page_range,
        "special_page":special_page,
        "page_overlap_char":page_overlap_char,
        "chunk_size":chunk_size,
        "chunk_overlap_char":chunk_overlap_char,
        "tolerance_rate":tolerance_rate,
        "min_chunk_size":min_chunk_size,
    })

    md_text = convert_to_md(path_file, page_range=page_range)
    if page_range is not None:
        start_page_index = page_range[0]
    else:
        start_page_index = 0

    match type_doc:
        case "sdg_evidence":
            metadata_document = extract_metadata(md_text, source)
            if (metadata_document['abstract'] or metadata_document['conclusion']) is None:
                result_extraction = extract_chunk(
                    text=md_text,
                    type_doc=type_doc,
                    source=source,
                    start_page_index=start_page_index,
                    special_page=special_page,
                    page_overlap_char=page_overlap_char,
                    chunk_size=chunk_size,
                    chunk_overlap_char=chunk_overlap_char,
                    tolerance_rate=tolerance_rate,
                    min_chunk_size=min_chunk_size
                )
            else:
                result_extraction = metadata_document

        case "sdg_knowledge":
            result_extraction = extract_chunk(
                text=md_text,
                type_doc=type_doc,
                source=source,
                start_page_index=start_page_index,
                special_page=special_page,
                page_overlap_char=page_overlap_char,
                chunk_size=chunk_size,
                chunk_overlap_char=chunk_overlap_char,
                tolerance_rate=tolerance_rate,
                min_chunk_size=min_chunk_size
            )
        case _:
            raise ValueError("Invalid Type of Document")
        
    


    if save_path != None:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result_extraction, f, ensure_ascii=False, indent=2)
    
    logger.debug("[END] : Extracting Document (total chars: %s)", len(str(result_extraction)))

    return result_extraction


def input_document(
        path_file:str,
        source:str = None,
        type_doc: Literal["sdg_evidence", "sdg_knowledge"] = "sdg_evidence",
        page_range: Optional[list[int]] = None,
        special_page: list[int] = [],
        page_overlap_char: int = 100,
        chunk_size: int = 500,
        chunk_overlap_char: int = 100,
        tolerance_rate: float = 0.5,
        min_chunk_size: int = 100,
):
    """
    Titik entry point untuk pemrosesan awal dokumen ke dalam sistem AI SDG.

    Fungsi ini bertindak sebagai orkestrator yang memanggil `extract_document`. Perbedaan utamanya adalah fungsi ini melakukan standarisasi output;
    jika hasil ekstraksi berupa potongan teks (chunks), maka akan secara otomatis dibungkus menjadi daftar objek `Document` agar kompatibel dengan modul pengolah teks selanjutnya.

    Args:
        path_file (str): Path lengkap menuju file dokumen yang akan diproses.
        source (str, optional): Nama instansi atau sumber asal dokumen. Defaults to None.
        type_doc (Literal["sdg_evidence", "sdg_knowledge"], optional): Klasifikasi dokumen untuk menentukan alur ekstraksi. "sdg_evidence" untuk bukti kinerja unit kerja, dan "sdg_knowledge" untuk referensi kriteria SDGs. Defaults to "sdg_evidence".

    Returns:
        Union[dict, List[Document]]: 
            - Jika hasil ekstraksi adalah metadata tunggal, mengembalikan `dict` berisi metadata (title, abstract, conclusion, dll).
            - Jika hasil ekstraksi adalah konten ter-chunk, mengembalikan `list` berisi objek `Document` (page_content & metadata).
    """
    logger.info("Preprocessing Document Input")
    result_extraction = extract_document(
        path_file=path_file,
        type_doc=type_doc,
        source=source,
        page_range=page_range,
        special_page=special_page,
        page_overlap_char=page_overlap_char,
        chunk_size=chunk_size,
        chunk_overlap_char=chunk_overlap_char,
        tolerance_rate=tolerance_rate,
        min_chunk_size=min_chunk_size,
    )

    if isinstance(result_extraction, list):
        documents: list[Document] = []

        for item in result_extraction:
            doc = Document(
                page_content=item['text'],
                metadata=item['metadata']
            )
            documents.append(doc)

        return documents
    
    return result_extraction