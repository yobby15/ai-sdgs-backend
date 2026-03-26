from app.utils import text_processing

import pymupdf4llm
import json
import logging
from typing import Literal

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

    result = convert_to_md(path_file, page_range=page_range)
    splitted_page = text_processing.split_and_clean_pages(result, special_page=special_page, overlap_chars=page_overlap_char, tolerance=round(tolerance_rate*page_overlap_char))
    chunked_content_per_page = [text_processing.chunk_text(content, chunk_size=chunk_size, overlap=chunk_overlap_char, tolerance=round(tolerance_rate*chunk_size), min_chunk_size=min_chunk_size) for content in splitted_page]
    clean_extracted = text_processing.pages_to_json_format(chunked_content_per_page, type_doc=type_doc, source=source)

    if save_path != None:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(clean_extracted, f, ensure_ascii=False, indent=2)
    
    logger.debug("[END] : Extracting Document (total chunk: %s)", str(len(clean_extracted)))

    return clean_extracted


def input_document(path_file, source):
    """
    **return format** (chunk_ids, texts, metadatas)
    """
    logger.info("Preprocessing Document Input")
    result_extraction = extract_document(
        path_file=path_file,
        type_doc="sdg_evidence",
        source=source
    )

    chunk_ids = []
    texts = []
    metadatas = []

    for item in result_extraction:
        chunk_ids.append(str(item["metadata"]["global_chunk_id"]))
        texts.append(item['text'])
        metadatas.append(item['metadata'])

    return chunk_ids, texts, metadatas