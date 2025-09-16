import os
from pathlib import Path
from dotenv import load_dotenv
from collections import Counter

from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

SOURCE_DOCS_PATH = "data/source_docs"
VECTOR_STORE_PATH = "data/vector_store"
EMBEDDING_MODEL = "thenlper/gte-small"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 300

def get_document_loaders():
    print("🔎 Configurando carregadores de documentos...")

    pdf_loader = DirectoryLoader(
        SOURCE_DOCS_PATH,
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader,
        show_progress=True,
        use_multithreading=False,
    )
    return [pdf_loader]


def process_and_index_documents():
    print("📂 Iniciando ingestão de documentos...")

    all_docs = []
    loaded_files = set()

    loaders = get_document_loaders()
    for loader in loaders:
        loader_name = type(loader.loader_cls).__name__
        print(f"➡️ Carregando documentos com {loader_name}...")

        try:
            loaded_docs = loader.load()
        except Exception as e:
            print(f"⚠️ Erro ao carregar com {loader_name}: {e}")
            continue

        for doc in loaded_docs:
            source_path = Path(doc.metadata.get("source", "unknown"))
            source_name = source_path.name
            file_extension = source_path.suffix.lstrip('.').lower()

            doc.metadata["source"] = source_name
            doc.metadata["type"] = file_extension

            if source_name not in loaded_files:
                print(f"📄 Documento carregado: {source_name} (tipo: {file_extension})")
                loaded_files.add(source_name)

        all_docs.extend(loaded_docs)

    if not all_docs:
        print("⚠️ Nenhum documento encontrado. Encerrando.")
        return

    print(f"\n✅ Total de {len(all_docs)} páginas/seções carregadas de {len(loaded_files)} arquivos.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = text_splitter.split_documents(all_docs)

    counts = Counter([c.metadata["source"] for c in chunks])
    print("\n📊 Chunks gerados por arquivo:")
    for src, n in counts.items():
        print(f"   - {src}: {n} chunks")

    print(f"✂️ Total de {len(chunks)} chunks gerados.")

    print(f"\n🧠 Criando embeddings com o modelo: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("📦 Criando e salvando o índice FAISS...")
    vector_store = FAISS.from_documents(chunks, embeddings)

    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
    vector_store.save_local(VECTOR_STORE_PATH)

    print("-" * 50)
    print("🎉 Processo de ingestão concluído com sucesso!")
    print(f"📍 Índice FAISS salvo em: {VECTOR_STORE_PATH}")
    print("-" * 50)

if __name__ == "__main__":
    process_and_index_documents()
