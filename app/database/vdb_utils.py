from langchain_core.vectorstores import VectorStore, InMemoryVectorStore
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.documents import Document

import logging
import uuid
from tqdm import tqdm

logger = logging.getLogger(name=__name__)

NAMESPACE_SDG = uuid.NAMESPACE_DNS

def add_data_to_vdb(
    vdb: VectorStore,
    documents: list[Document] = None,
    vectors: list[list] = None,
    texts: list[str] = None,
    metadatas: list[dict] = None,
    ids: list[str] = None
):
    """
    Menambahkan pengetahuan terkait SDG ke dalam Vector Database.

    Fungsi ini mendukung dua metode input dengan urutan prioritas:
    1. Menggunakan objek `documents` (Prioritas Utama).
    2. Menggunakan pasangan `texts` dan `metadatas` jika `documents` tidak tersedia.

    Args:
        vdb (VectorStore): Instance vector database yang digunakan.
        documents (list[Document], optional): List objek Document. Jika diisi, parameter 
            `texts` dan `metadatas` akan diabaikan.
        texts (list[str], optional): List teks mentah yang akan di-embed.
        metadatas (list[dict], optional): List metadata untuk tiap teks. Harus memiliki 
            panjang yang sama dengan `texts`.
        ids (list[str], optional): List ID unik untuk tiap entry. Memudahkan update/delete nantinya.

    Returns:
        List[str]: List berisi ID dari dokumen yang berhasil dimasukkan ke database.
    """

    logger.debug("[START] : Adding data to VectoreDatabase (input parameters: %s)", {
        "type_vdb":type(vdb),
        "type_documents":type(documents),
        "type_texts":type(texts),
        "type_metadatas":type(metadatas),
        "type_ids":type(ids)
    })
    if documents is None:
        if (texts and metadatas) is not None:
            if len(texts) != len(metadatas):
                logger.error(f"Panjang texts dan metadata tidak sama! texts: {len(texts)} != metadatas: {len(metadatas)}")
                raise ValueError("Panjang texts dan metadata tidak sama!")
            
            documents = []
            for text, metadata in zip(texts, metadatas):
                doc = Document(
                    page_content=text,
                    metadata=metadata
                )
                documents.append(doc)
        else:
            logger.error(f"documents and (texts & metadatas) both is None!")
            raise ValueError("documents and (texts & metadatas) both is None!")
            
        
    if ids is None:
        ids = []
        for i in range(len(documents)):
            unique_id = str(uuid.uuid5(NAMESPACE_SDG, f"{documents[i].metadata}_{documents[i].page_content}"))
            ids.append(unique_id)

    elif len(ids) != len(documents):
        logger.warning("Total ids tidak sama dengan total data yang dimasukkan! (otomatis dilakukan pengurangan atau penambahan)")
        if len(ids) < len(documents):
            selisih = len(documents) - len(ids)
            for i in range(selisih):
                unique_id = str(uuid.uuid5(NAMESPACE_SDG, f"{documents[len(ids)].metadata}_{documents[len(ids)].page_content}"))
                ids.append(unique_id)
        else:
            ids = ids[:len(documents)]

    logger.debug("Total data yang dimasukkan ke Database %s", len(documents))
    if vectors is not None:
        if len(vectors) != len(documents):
            logger.error("Total data vector embedding tidak sama dengan total documents")
            raise ValueError("Total data vector embedding tidak sama dengan total documents!")
        
        if isinstance(vdb, SupabaseVectorStore):
            vdb.add_vectors(vectors=vectors, documents=documents, ids=ids)

    BATCH_SIZE: int = 10
    for i in tqdm(range(0, len(documents), BATCH_SIZE)):
        vdb.add_documents(documents=documents[i:i+BATCH_SIZE], ids=ids[i:i+BATCH_SIZE])
    logger.debug("[END] : Adding data to VectoreDatabase")

    return ids
