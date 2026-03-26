import re
import json
from typing import Literal

PAGE_PATTERN = re.compile(r"--- end of page\.page_number=\d+ ---", re.IGNORECASE)
BOLD_BR_PATTERN = re.compile(r"\*\*(.*?)\*\*\s*<br\s*/?>\s*\*\*(.*?)\*\*", re.IGNORECASE | re.DOTALL)


def repair_llm_json(text: str) -> dict:
    """
    Parse JSON dari output LLM dan memperbaiki kesalahan umum.
    """
    
    if not isinstance(text, str):
        raise TypeError("Input harus berupa string")

    text = text.strip()

    # 1. Hapus markdown code block
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```", "", text)
    text = re.sub(r"```$", "", text)

    # 2. Ambil bagian JSON saja (jika ada teks lain)
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1:
        text = text[start:end+1]

    # 3. Hapus trailing comma
    text = re.sub(r",(\s*[}\]])", r"\1", text)

    # 4. Coba parse langsung
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 5. Repair tambahan (quote keys jika hilang)
    text = re.sub(r'(\w+):', r'"\1":', text)

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON masih tidak valid setelah repair:\n{text}") from e
    

def find_chunk_splitter(text: str, start: int, target: int, tolerance: int) -> int:
    """
    Mencari posisi split terbaik berdasarkan prioritas:
    paragraph → newline → sentence → fallback
    """

    n = len(text)
    lower = max(start, target - tolerance)
    upper = min(n, target + tolerance)
    region = text[lower:upper]

    idx = region.rfind("\n\n", 0, target - lower)
    if idx != -1:
        return lower + idx + 2

    idx = region.find("\n\n", target - lower)
    if idx != -1:
        return lower + idx + 2

    idx = region.rfind("\n", 0, target - lower)
    if idx != -1:
        return lower + idx + 1

    idx = region.find("\n", target - lower)
    if idx != -1:
        return lower + idx + 1

    sentence_pattern = r"[.!?]\s"

    matches = list(re.finditer(sentence_pattern, region))

    best = None
    best_dist = None

    for m in matches:
        pos = lower + m.end()
        dist = abs(pos - target)
        if best is None or dist < best_dist:
            best = pos
            best_dist = dist

    if best is not None:
        return best

    return min(target, n)


def clean_page_text(text: str) -> str:
    """
    Membersihkan noise pada text tanpa menghapus newline.
    """

    if not text:
        return text

    # Gabungkan bold yang dipisahkan <br>
    while BOLD_BR_PATTERN.search(text):
        text = BOLD_BR_PATTERN.sub(
            lambda m: f"**{m.group(1).strip()} {m.group(2).strip()}**",
            text
        )

    # Hapus <br> saja (tanpa menyentuh \n)
    text = re.sub(r"<br\s*/?>", "", text, flags=re.IGNORECASE)

    return text.strip()


def split_and_clean_pages(
    text: str,
    overlap_chars: int = 100,
    tolerance: int = 200,
    special_page: list[int] = []
) -> list[str]:
    """
    Memisahkan teks berdasarkan marker halaman
    dan membersihkan setiap halaman.
    """

    if not text:
        return []

    pages = re.split(PAGE_PATTERN, text)

    cleaned_pages = [
        clean_page_text(page)
        for page in pages
        if page.strip()
    ]

    cleaned_pages = add_page_overlap(
        pages=cleaned_pages,
        overlap_chars=overlap_chars,
        tolerance=tolerance,
        special_page=special_page
    )
    return cleaned_pages


def add_page_overlap(
    pages: list[str],
    overlap_chars: int = 100,
    tolerance: int = 200,
    special_page: list[int] = []
) -> list[str]:
    """
    Menambahkan overlap antar halaman.
    Overlap diambil dari akhir halaman sebelumnya
    tetapi tetap dipotong pada boundary terbaik.
    """

    if not pages:
        return pages

    new_pages = [pages[0]]
    special_page = set(special_page)

    for i in range(1, len(pages)):
        if (i in special_page) or ((i-1) in special_page):
            if i in special_page:
                print(pages[i])
            new_pages.append(pages[i])
            continue

        prev_page = pages[i - 1]
        curr_page = pages[i]

        target = len(prev_page) - overlap_chars

        if target <= 0:
            overlap = prev_page
        else:
            split = find_chunk_splitter(prev_page, 0, target, tolerance)
            overlap = prev_page[split:]

        new_pages.append(overlap + curr_page)

    return new_pages


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
    tolerance: int = 200,
    min_chunk_size: int = 100,
) -> list[str]:
    """
    Smart chunking untuk RAG pipeline.

    Features:
    - paragraph aware
    - newline aware
    - sentence aware
    - overlap stabil
    - menghindari tiny chunk
    """

    chunks = []
    n = len(text)

    start = 0

    while start < n:

        target = start + chunk_size

        if target >= n:
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        split = find_chunk_splitter(text, start, target, tolerance)

        chunk = text[start:split].strip()

        if len(chunk) < min_chunk_size and chunks:
            # gabungkan dengan chunk sebelumnya
            chunks[-1] += "\n" + chunk
        else:
            chunks.append(chunk)

        # hitung overlap
        overlap_target = split - overlap

        if overlap_target <= start:
            start = split
        else:
            start = find_chunk_splitter(text, start, overlap_target, tolerance)

    return chunks


def pages_to_json_format(
        pages: list[list[str]],
        source: str,
        type_doc: Literal["sdg_evidence", "sdg_knoledge"] = "sdg_evidence",
        ) -> list[dict]:
    """
    Mengubah data [page_list[chunk]] menjadi format JSON
    yang siap digunakan oleh vector database.
    """

    results = []
    global_chunk_id = 0

    for page_idx, chunks in enumerate(pages):

        page_number = page_idx + 1

        for chunk_idx, chunk_text in enumerate(chunks):

            item = {
                "text": chunk_text,
                "metadata": {
                    "page": int(page_number),
                    "page_chunk_id": int(chunk_idx),
                    "global_chunk_id":int(global_chunk_id),
                    "source": source,
                    "type":type_doc,
                }
            }

            results.append(item)

            global_chunk_id += 1

    return results