"""
seed_sdg.py
-----------
Script sekali jalan untuk populate tabel `sdg_vectorstore` di Supabase
dengan data indikator dari dokumen SDG Methodology PDF.

Cara pakai:
    python seed_sdg.py --pdf "path/to/Sustainability Impact Ratings Methodology 2026.pdf"

Opsi tambahan:
    --page-start     Halaman mulai ekstraksi (default: 10, halaman sebelumnya cover/daftar isi)
    --page-end       Halaman akhir ekstraksi (default: None = sampai akhir)
    --batch-size     Jumlah chunk per batch embed & insert ke Supabase (default: 32)
    --clear          Hapus semua data lama di tabel sebelum insert
    --dry-run        Jalankan ekstraksi & embedding tanpa insert ke Supabase
"""

import argparse
import logging
import os
import sys
import time
import uuid
from datetime import timedelta

import faiss
import numpy as np
from dotenv import load_dotenv

from app.core.input_doc import extract_document
from app.database.supabase_service import supabase_init
from app.infrastructure.embedding_service import embedding_init

load_dotenv()

# ---------------------------------------------------------------------------
# Warna & tampilan terminal
# ---------------------------------------------------------------------------

class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"
    GREY    = "\033[90m"
    BLUE    = "\033[94m"


class PrettyFormatter(logging.Formatter):
    LEVEL_STYLES = {
        logging.DEBUG:    f"{C.GREY}  DBG{C.RESET}",
        logging.INFO:     f"{C.CYAN} INFO{C.RESET}",
        logging.WARNING:  f"{C.YELLOW} WARN{C.RESET}",
        logging.ERROR:    f"{C.RED}  ERR{C.RESET}",
        logging.CRITICAL: f"{C.RED}{C.BOLD} CRIT{C.RESET}",
    }

    def format(self, record: logging.LogRecord) -> str:
        level = self.LEVEL_STYLES.get(record.levelno, record.levelname)
        timestamp = self.formatTime(record, "%H:%M:%S")
        msg = record.getMessage()
        return f"{C.GREY}{timestamp}{C.RESET} {level}  {msg}"


handler = logging.StreamHandler()
handler.setFormatter(PrettyFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])

for _noisy in ["httpcore", "httpx", "hpack", "sentence_transformers", "transformers"]:
    logging.getLogger(_noisy).setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


def print_header(title: str) -> None:
    width = 60
    print(f"\n{C.BOLD}{C.BLUE}{'─' * width}{C.RESET}")
    print(f"{C.BOLD}{C.WHITE}  {title}{C.RESET}")
    print(f"{C.BOLD}{C.BLUE}{'─' * width}{C.RESET}\n")


def print_step(step: int, total: int, label: str) -> None:
    print(f"\n{C.BOLD}{C.CYAN}[{step}/{total}]{C.RESET} {C.WHITE}{label}{C.RESET}")


def print_progress(current: int, total: int, label: str, start_time: float) -> None:
    pct    = current / total
    filled = int(30 * pct)
    bar    = f"{'█' * filled}{'░' * (30 - filled)}"
    elapsed = time.time() - start_time
    eta     = (elapsed / pct - elapsed) if pct > 0 else 0
    print(
        f"\r  {C.CYAN}{bar}{C.RESET} "
        f"{C.BOLD}{current:>4}/{total}{C.RESET} "
        f"{C.GREY}| {label} "
        f"| elapsed {timedelta(seconds=int(elapsed))} "
        f"| ETA {timedelta(seconds=int(eta))}{C.RESET}",
        end="",
        flush=True,
    )
    if current >= total:
        print()


def print_ok(msg: str) -> None:
    print(f"  {C.GREEN}{C.BOLD}✓{C.RESET}  {msg}")


def print_warn(msg: str) -> None:
    print(f"  {C.YELLOW}{C.BOLD}⚠{C.RESET}  {msg}")


def print_err(msg: str) -> None:
    print(f"\n  {C.RED}{C.BOLD}✗  {msg}{C.RESET}\n")


def print_summary(stats: dict) -> None:
    width = 60
    print(f"\n{C.BOLD}{C.BLUE}{'─' * width}{C.RESET}")
    print(f"{C.BOLD}{C.WHITE}  RINGKASAN{C.RESET}")
    print(f"{C.BOLD}{C.BLUE}{'─' * width}{C.RESET}")
    for key, val in stats.items():
        print(f"  {C.GREY}{key:<26}{C.RESET} {val}")
    print(f"{C.BOLD}{C.BLUE}{'─' * width}{C.RESET}\n")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def embed_chunks(texts: list[str], embeddings, batch_size: int = 32) -> list[list[float]]:
    total      = len(texts)
    all_vectors = []
    start_time = time.time()

    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        try:
            batch_vectors = embeddings.embed_documents(batch)
        except Exception as e:
            print_err(f"Gagal embed batch {i}–{i + len(batch)}: {e}")
            raise

        batch_np = np.array(batch_vectors).astype("float32")
        faiss.normalize_L2(batch_np)
        all_vectors.extend(batch_np.tolist())
        print_progress(min(i + batch_size, total), total, "embedding", start_time)

    return all_vectors


def insert_batch(supabase, rows: list[dict]) -> None:
    response = supabase.table("sdg_vectorstore").insert(rows).execute()
    if hasattr(response, "error") and response.error:
        raise RuntimeError(f"Supabase insert error: {response.error}")


def clear_table(supabase) -> None:
    print_warn("Menghapus semua data lama di sdg_vectorstore...")
    supabase.table("sdg_vectorstore").delete().neq("id", 0).execute()
    print_ok("Tabel berhasil dikosongkan.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def seed(
    pdf_path:   str,
    page_start: int  = 10,
    page_end:   int  = None,
    batch_size: int  = 32,
    clear:      bool = False,
    dry_run:    bool = False,
) -> None:

    total_steps  = 4 if not dry_run else 3
    overall_start = time.time()

    print_header("SDG Vectorstore Seeder")
    if dry_run:
        print_warn("Mode DRY RUN aktif — tidak ada data yang dikirim ke Supabase.\n")

    print_step(1, total_steps, "Validasi file & konfigurasi")

    if not os.path.exists(pdf_path):
        print_err(f"File tidak ditemukan: {pdf_path}")
        sys.exit(1)

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
    if not dry_run and (not supabase_url or not supabase_key):
        print_err("SUPABASE_URL dan SUPABASE_SERVICE_KEY wajib ada di .env")
        sys.exit(1)

    source     = os.path.basename(pdf_path)
    page_range = [page_start, page_end] if page_end else None

    logger.info("File      : %s", pdf_path)
    logger.info("Source    : %s", source)
    logger.info("Halaman   : %s – %s", page_start, page_end or "akhir")
    logger.info("Batch size: %d", batch_size)
    print_ok("Validasi selesai.")

    print_step(2, total_steps, "Ekstraksi & chunking dokumen PDF")
    t0 = time.time()

    try:
        chunks = extract_document(
            path_file=pdf_path,
            type_doc="sdg_knowledge",
            source=source,
            page_range=page_range,
        )
    except Exception as e:
        print_err(f"Gagal mengekstrak dokumen: {e}")
        raise

    print_ok(f"{len(chunks)} chunk berhasil diekstrak  ({time.time() - t0:.1f}s)")

    print_step(3, total_steps, f"Embedding {len(chunks)} chunk  (batch size: {batch_size})")
    t0 = time.time()

    logger.info("Memuat embedding model...")
    try:
        embeddings = embedding_init(model_name="microsoft/harrier-oss-v1-0.6b", type_run="huggingface_inference")
    except Exception as e:
        print_err(f"Gagal memuat embedding model: {e}")
        raise

    try:
        vectors = embed_chunks(texts=[c["text"] for c in chunks], embeddings=embeddings, batch_size=batch_size)
    except Exception:
        print_err("Proses embedding gagal. Lihat pesan error di atas.")
        sys.exit(1)

    print_ok(f"Semua chunk berhasil di-embed  ({time.time() - t0:.1f}s)")

    if dry_run:
        print_summary({
            "Mode"             : f"{C.YELLOW}DRY RUN{C.RESET}",
            "File"             : source,
            "Total chunk"      : str(len(chunks)),
            "Contoh chunk [0]" : chunks[0]["text"][:70] + "...",
            "Contoh vektor [0]": str([round(v, 4) for v in vectors[0][:4]]) + "...",
        })
        return

    print_step(4, total_steps, "Insert ke Supabase")

    supabase = supabase_init(supabase_url=supabase_url, supabase_service_key=supabase_key)
    logger.info("Koneksi Supabase berhasil.")

    if clear:
        clear_table(supabase)

    total    = len(chunks)
    inserted = 0
    failed   = 0
    t0       = time.time()

    for i in range(0, total, batch_size):
        batch_chunks  = chunks[i : i + batch_size]
        batch_vectors = vectors[i : i + batch_size]

        rows = [
            {
                "id": str(uuid.uuid4()),
                "content"  : chunk["text"],
                "metadata" : chunk["metadata"],
                "embedding": vector,
            }
            for chunk, vector in zip(batch_chunks, batch_vectors)
        ]

        try:
            insert_batch(supabase, rows)
            inserted += len(rows)
        except Exception as e:
            failed += len(rows)
            logger.error("Batch %d–%d gagal diinsert: %s", i, i + len(rows), e)

        print_progress(i + len(rows), total, "inserting", t0)

    elapsed = time.time() - overall_start

    if failed == 0:
        status = f"{C.GREEN}{C.BOLD}SUKSES{C.RESET}"
    else:
        status = f"{C.YELLOW}{C.BOLD}SELESAI — {failed} chunk gagal{C.RESET}"

    print_summary({
        "Status"             : status,
        "File"               : source,
        "Total chunk"        : str(total),
        "Berhasil di-insert" : f"{C.GREEN}{inserted}{C.RESET}",
        "Gagal"              : f"{C.RED}{failed}{C.RESET}" if failed else f"{C.GREEN}0{C.RESET}",
        "Total waktu"        : str(timedelta(seconds=int(elapsed))),
    })

    if failed > 0:
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Populate tabel sdg_vectorstore di Supabase dari PDF SDG Methodology."
    )
    parser.add_argument("--pdf",        required=True,        help="Path ke file PDF SDG Methodology")
    parser.add_argument("--page-start", type=int, default=10, help="Halaman mulai ekstraksi (default: 10)")
    parser.add_argument("--page-end",   type=int, default=None, help="Halaman akhir ekstraksi (default: sampai akhir)")
    parser.add_argument("--batch-size", type=int, default=32,  help="Jumlah chunk per batch embed & insert (default: 32)")
    parser.add_argument("--clear",      action="store_true",   help="Hapus semua data lama sebelum insert")
    parser.add_argument("--dry-run",    action="store_true",   help="Jalankan tanpa insert ke Supabase")

    args = parser.parse_args()

    seed(
        pdf_path   = args.pdf,
        page_start = args.page_start,
        page_end   = args.page_end,
        batch_size = args.batch_size,
        clear      = args.clear,
        dry_run    = args.dry_run,
    )