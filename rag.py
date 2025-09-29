import os
import hashlib
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from config import DOCUMENTS_DIR, CHROMA_PATH

# Используем локальную embedding-модель (не зависит от OpenRouter!)
embedding_function = HuggingFaceEmbeddings(
    model_name="cointegrated/rubert-tiny2",
    model_kwargs={'device': 'cpu'}
)

def get_file_hash(filepath: str) -> str:
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def load_documents() -> list[Document]:
    docs = []
    documents_path = Path(DOCUMENTS_DIR)
    if not documents_path.exists():
        documents_path.mkdir(parents=True, exist_ok=True)
        return docs

    for file_path in documents_path.glob("*"):
        if file_path.is_file():
            ext = file_path.suffix.lower()
            if ext == ".pdf":
                try:
                    loader = PyPDFLoader(str(file_path))
                    docs.extend(loader.load())
                except Exception as e:
                    print(f"⚠️ Не удалось загрузить PDF {file_path}: {e}")
            elif ext == ".txt":
                try:
                    loader = TextLoader(str(file_path), encoding="utf-8")
                    docs.extend(loader.load())
                except Exception as e:
                    print(f"⚠️ Не удалось загрузить TXT {file_path}: {e}")
    return docs

def get_processed_files_hash() -> str:
    files = sorted(Path(DOCUMENTS_DIR).glob("*"))
    file_hashes = []
    for f in files:
        if f.is_file():
            file_hashes.append(get_file_hash(str(f)))
    combined = "".join(file_hashes)
    return hashlib.md5(combined.encode()).hexdigest()

# Кэширование векторной БД
_vectorstore = None
_last_hash = None

def get_or_create_vectorstore() -> Chroma:
    global _vectorstore, _last_hash
    current_hash = get_processed_files_hash()

    if _vectorstore is None or _last_hash != current_hash:
        print("🔄 Обнаружены новые или изменённые документы. Пересоздаю векторную БД...")
        documents = load_documents()
        if not documents:
            print("⚠️ Папка documents пуста. Векторная БД не будет создана.")
            # Создаём пустую коллекцию, чтобы избежать ошибок
            _vectorstore = Chroma(
                embedding_function=embedding_function,
                persist_directory=CHROMA_PATH
            )
            _last_hash = current_hash
            return _vectorstore

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                       chunk_overlap=150,
                                                       separators=["\n\n", "\n", ". ", " ", ""])
        chunks = text_splitter.split_documents(documents)
        _vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_function,
            persist_directory=CHROMA_PATH
        )
        _last_hash = current_hash
        print(f"✅ Векторная БД создана из {len(chunks)} чанков.")
    return _vectorstore