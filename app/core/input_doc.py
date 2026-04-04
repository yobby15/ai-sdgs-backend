from app.utils import text_processing

import pymupdf4llm
import json
import logging
from typing import Literal
import re

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
    Funtion untuk mengekstrak metadata dari dokumen artikel atau paper Dosen. Metadata yang diekstrak [title, abstract, conclusion].
    - `text`: text dokumen dengan format markdown,
    - `doc_name`: nama dokumen

    **return** dictionary berisi metadata dokumen.
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

def extract_document(
        path_file: str,
        type_doc: Literal["sdg_evidence", "sdg_knowledge"],
        source: str,
        save_path: str = None,
        page_range: list[int, int] = None,
        special_page: list[int] = [],
        page_overlap_char: int = 100,
        chunk_size: int = 500,
        chunk_overlap_char: int = 100,
        tolerance_rate: float = 0.5,
        min_chunk_size: int = 100,
):
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

    path_file = path_file
    type_doc = type_doc
    source = source

    page_range = page_range
    special_page = special_page
    page_overlap_char = page_overlap_char

    chunk_size = chunk_size
    chunk_overlap_char = chunk_overlap_char
    tolerance_rate = tolerance_rate
    min_chunk_size = min_chunk_size

    md_text = convert_to_md(path_file, page_range=page_range)
    metadata_document = extract_metadata(md_text, source)

    # splitted_page = text_processing.split_and_clean_pages(md_text, special_page=special_page, overlap_chars=page_overlap_char, tolerance=round(tolerance_rate*page_overlap_char))
    # chunked_content_per_page = [text_processing.chunk_text(content, chunk_size=chunk_size, overlap=chunk_overlap_char, tolerance=round(tolerance_rate*chunk_size), min_chunk_size=min_chunk_size) for content in splitted_page]
    # clean_extracted = text_processing.pages_to_json_format(chunked_content_per_page, type_doc=type_doc, source=source)

    if save_path != None:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(metadata_document, f, ensure_ascii=False, indent=2)
    
    logger.debug("[END] : Extracting Document (total chars: %s)", len(str(metadata_document)))

    return metadata_document


def input_document(path_file, source):
    """
    **return**  metadata dari dokumen dengan key {'document_name', 'title', 'abstract', 'conclusion'}
    """
    logger.info("Preprocessing Document Input")
    result_extraction = extract_document(
        path_file=path_file,
        type_doc="sdg_evidence",
        source=source
    )

    # chunk_ids = []
    # texts = []
    # metadatas = []

    # for i, (key, value) in enumerate(result_extraction.items()):
    #     if key == "document_name":
    #         continue

    #     chunk_ids.append(str(i))
    #     texts.append(value)
    #     metadatas.append({
    #         "source": result_extraction["document_name"],
    #         "type_text":key,
    #     })

    return result_extraction
    # return chunk_ids, texts, metadatas