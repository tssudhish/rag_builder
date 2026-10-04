from rag_builder.ingestion.loaders import load_document
from rag_builder.ingestion.splitter import TextSplitter

def test_ingestion():
    test_files = ["test_doc.txt", "test_doc.md", "test_doc.html"]
    splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
    
    for file in test_files:
        print(f"Testing {file}...")
        try:
            doc = load_document(file)
            print(f"  Loaded content length: {len(doc.content)}")
            print(f"  Metadata: {doc.metadata}")
            
            chunks = splitter.split_document(doc)
            print(f"  Split into {len(chunks)} chunks")
            for i, chunk in enumerate(chunks):
                print(f"    Chunk {i}: {chunk.content[:50]}... (offset: {chunk.char_offset})")
            print("-" * 20)
        except Exception as e:
            print(f"  Error testing {file}: {e}")

if __name__ == "__main__":
    test_ingestion()
