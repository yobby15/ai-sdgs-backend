import re
import json
from typing import Literal
import warnings

PAGE_PATTERN = re.compile(r"--- end of page\.page_number=\d+ ---", re.IGNORECASE)
BOLD_BR_PATTERN = re.compile(r"\*\*(.*?)\*\*\s*<br\s*/?>\s*\*\*(.*?)\*\*", re.IGNORECASE | re.DOTALL)
FORMATTED_PATTERN = re.compile(r"\*\*==>.*?<==\*\*")
ABSTRACT_WORDS_PATTERN = [
    r"a\s*b\s*s\s*t\s*r\s*a\s*c\s*t",
    r"a\s*b\s*s\s*t\s*r\s*a\s*k",
    r"i\s*n\s*t\s*i\s*s\s*a\s*r\s*i"
]
KEYWORD_WORDS_PATTERN = [
    r"k\s*e\s*y\s*w\s*o\s*r\s*d[s]*",
    r"k\s*a\s*t\s*a\s*.?\s*k\s*u\s*n\s*c\s*i"
]
INTRO_WORDS_PATTERN = [
    r"i\s*n\s*t\s*r\s*o\s*d\s*u\s*c\s*t\s*i\s*o\s*n[s]*",
    r"p\s*e\s*n\s*d\s*a\s*h\s*u\s*l\s*u\s*a\s*n"
]

CONCLUSION_WORDS_PATTERN = [
    r"c\s*o\s*n\s*c\s*l\s*u\s*s\s*i\s*o\s*n\s*[s]*",
    r"(k\s*e\s*)?s\s*i\s*m\s*p\s*u\s*l\s*a\s*n",
    r"p\s*e\s*n\s*u\s*t\s*u\s*p"
]

DISCUSSION_WORDS_PATTERN = [
    r"d\s*i\s*s\s*c\s*u\s*s\s*s\s*i\s*o\s*n[s]?",
    r"d\s*i\s*s\s*k\s*u\s*s\s*i"
]

REFERENCE_WORDS_PATTERN = [
    r"r\s*e\s*f\s*e\s*r\s*e\s*n\s*c\s*e[s]?",
    r"r\s*e\s*f\s*e\s*r\s*e\s*n\s*s\s*i",
    r"d\s*a\s*f\s*t\s*a\s*r\s*(p\s*u\s*s\s*t\s*a\s*k\s*a|a\s*c\s*u\s*a\s*n|r\s*u\s*j\s*u\s*k\s*a\s*n)",
    r"p\s*u\s*s\s*t\s*a\s*k\s*a\s*a\s*c\s*u\s*a\s*n",
    r"b\s*l\s*i\s*b\s*l\s*i\s*o\s*g\s*r\s*a\s*f\s*i"
]

ABSTRACT_PATTERN = rf"(?i)\b({'|'.join(ABSTRACT_WORDS_PATTERN)})\b"
CONCLUSION_PATTERN = rf"(?im)^##.*?\b({'|'.join(CONCLUSION_WORDS_PATTERN)})\b.*"
DISCUSSION_PATTERN = rf"(?im)^##.*?\b({'|'.join(DISCUSSION_WORDS_PATTERN)})\b.*"
REFERENCE_PATTERN = rf"(?im)^##.*?\b({'|'.join(REFERENCE_WORDS_PATTERN)})\b.*"
END_ABSTRACT_PATTERN = rf"(?i)\b({'|'.join(KEYWORD_WORDS_PATTERN+INTRO_WORDS_PATTERN)}).*"


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

    # Hapus format noise
    text = re.sub(FORMATTED_PATTERN, "", text)

    return text.strip()


def split_and_clean_pages(
    text: str,
    add_overlap: bool=True,
    overlap_chars: int = 100,
    tolerance: int = 200,
    special_page: list[int] = []
) -> list[str]:
    """
    Memisahkan teks berdasarkan marker halaman serta membersihkan setiap halaman.
    - `text`: isi dokumen,
    - `add_overlap`: menambahkan overlap halaman,
    - `overlap_chars`: jumlah perkiraan karakter overlap,
    - `tolerance`: toleransi total karakter dari pemotongan chunk,
    - `special_page`: halaman yang tidak ditambahkan overlap
    
    **return** list berisi string setiap halaman
    """

    if not text:
        return []

    pages = re.split(PAGE_PATTERN, text)

    cleaned_pages = [
        clean_page_text(page)
        for page in pages
        if page.strip()
    ]

    if add_overlap:
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


def clean_markdown(text: str) -> str:
    """
    Membersihkan format markdown menjadi plain text.
    """

    try:
        # 1) Hapus fenced code block
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

        # 2) Hapus inline code
        text = re.sub(r"`([^`]*)`", r"\1", text)

        # 3) Convert image markdown -> alt text saja
        text = re.sub(r"!\[(.*?)\]\(.*?\)", r"\1", text)

        # 4) Convert link markdown -> text saja
        text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)

        # 5) Hapus heading markdown (#, ##, ###)
        text = re.sub(r"(?m)^#{1,6}\s*", "", text)

        # 6) Hapus blockquote
        text = re.sub(r"(?m)^>\s*", "", text)

        # 7) Hapus bullet / numbered list marker
        text = re.sub(r"(?m)^\s*[-*+]\s+", "", text)
        text = re.sub(r"(?m)^\s*\d+\.\s+", "", text)

        # 8) Hapus bold / italic / underline markdown
        text = re.sub(r"(\*\*|\*|__|_)(.*?)\1", r"\2", text)

        # 9) Hapus horizontal rule
        text = re.sub(r"(?m)^[-*_]{3,}\s*$", "", text)

        # 10) Rapikan whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
    except Exception as e:
        warnings.warn(f"Tidak bisa membersihkan markdown pada text\n{e}")

    return text.strip()


def _extract_section(full_text: str, start_section_pattern: str, end_section_pattern: str) -> str | None:
    """
    Helper untuk mengambil isi sebuah section.
    """
    start_match = re.search(start_section_pattern, full_text)

    if not start_match:
        return None

    start_index = start_match.end()

    end_match = re.search(end_section_pattern, full_text[start_index:])

    if end_match:
        content = full_text[start_index : start_index + end_match.start()]
    else:
        return None

    clean_content = content.strip()
    return clean_content if clean_content else None


# =============== ABSTRACT EXTRACTION FUNCTION ===============
def extract_abstract(text: str) -> str:
    """
    Ekstraksi abstract dengan 2 layer:
    1. Stop saat section berikutnya muncul
    2. Fallback split paragraf
    """

    for pattern in ABSTRACT_WORDS_PATTERN:
        start_pattern = rf"(?im)\b{pattern}\b"
        abstract_candidate = _extract_section(text, start_section_pattern=start_pattern, end_section_pattern=END_ABSTRACT_PATTERN)
        abstract_candidate = abstract_candidate.lstrip(" :\n\r\t")
        if len(abstract_candidate) > 150:
            return abstract_candidate


    blocks = re.split(r"\n\s*\n", text)
    for block in blocks:
        clean_block = block.strip()
        if len(clean_block) > 300:
            return clean_block

    return None


# =============== TITLE EXTRACTION FUNCTION ===============
def _filter_candidates_before_abstract(candidates: list[str]) -> list[str]:
    """
    Hanya ambil kandidat sebelum bagian abstract.
    """
    filtered = []

    for candidate in candidates:
        clean = candidate.lower().strip()

        if re.search(ABSTRACT_PATTERN, clean):
            break

        filtered.append(candidate)
    
    return filtered

def _score_title_candidates(candidates: list[str]) -> str | None:
    scored_candidates = []

    for item in candidates:
        text = (
            item.replace("*", "")
            .replace("_", "")
            .replace("\n", " ")
            .strip()
        )

        if len(text) < 30:
            continue

        score = 0

        # panjang ideal
        if 40 <= len(text) <= 300:
            score += 50
        elif len(text) > 300:
            score -= 20

        # noise keywords
        noise_keywords = ["@", "http", "vol.", "issn", "penulis", "jurnal", "[", "]", "journal"]
        if any(word in text.lower() for word in noise_keywords):
            score -= 100
        else:
            score += 1

        scored_candidates.append({
            "text": text,
            "score": score
        })

    scored_candidates.sort(key=lambda x: x["score"], reverse=True)
    # print(scored_candidates)
    if scored_candidates and scored_candidates[0]["score"] > 0:
        return scored_candidates[0]["text"]

    return None

def extract_title(full_text: str) -> str:

    heading_candidates = re.findall(
        r"(?m)^(#[#]?\s*.*?)\s*$",
        full_text
    )
    heading_candidates = _filter_candidates_before_abstract(heading_candidates)
    best_title = _score_title_candidates(heading_candidates)
    if best_title:
        return best_title


    warnings.warn("Tidak ada header yang cocok menjadi title!\nMengambil semua baris sebagai title candidates")
    block_candidates = [
        block.strip()
        for block in full_text.split("\n\n")
        if block.strip()
    ]

    block_candidates = _filter_candidates_before_abstract(block_candidates)
    best_title = _score_title_candidates(block_candidates)
    if best_title:
        return best_title

    return None


# =============== CONDLUSION EXTRACTION FUNCTION ===============
def extract_conclusion(full_text: str) -> str:
    """
    Multi-layer conclusion extraction:
    1. conclusion / kesimpulan / simpulan / penutup
    2. fallback discussion / diskusi
    """

    result = _extract_section(
        full_text,
        start_section_pattern= CONCLUSION_PATTERN,
        end_section_pattern= REFERENCE_PATTERN
    )

    if result:
        return result

    result = _extract_section(
        full_text,
        start_section_pattern= DISCUSSION_PATTERN,
        end_section_pattern= REFERENCE_PATTERN
    )

    if result:
        return result

    return None