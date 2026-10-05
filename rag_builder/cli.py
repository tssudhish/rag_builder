import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Optional, List, Tuple

from rag_builder.ingestion.loaders import load_document
from rag_builder.ingestion.splitter import TextSplitter
from rag_builder.extraction.ollama_client import OllamaClient
from rag_builder.extraction.triplets import TripletExtractor
from rag_builder.storage.base import BaseGraphStorage, Triplet as StorageTriplet
from rag_builder.storage.memory_adapter import MemoryGraphStorage
from rag_builder.rag.retriever import GraphRetriever
from rag_builder.rag.generator import ResponseGenerator
from rag_builder.rag.query_generator import QueryGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger("rag_builder_cli")

def build_graph(file_path: str, model: str) -> BaseGraphStorage:
    """Pipeline to load a document, extract triplets, and build a graph."""
    logger.info(f"Loading document: {file_path}")
    doc = load_document(file_path)
    
    logger.info("Splitting document into chunks...")
    splitter = TextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_document(doc)
    
    logger.info(f"Extracting triplets using model {model}...")
    client = OllamaClient()
    extractor = TripletExtractor(client, model=model)
    storage = MemoryGraphStorage()
    
    total_triplets = 0
    for i, chunk in enumerate(chunks):
        triplets = extractor.extract(chunk.content)
        for t in triplets:
            storage.insert_triplet(StorageTriplet(
                subject=t.subject,
                predicate=t.predicate,
                object=t.object
            ))
            total_triplets += 1
        if (i + 1) % 5 == 0:
            logger.info(f"Processed {i + 1}/{len(chunks)} chunks...")

    logger.info(f"Graph built with {total_triplets} triplets.")
    return storage

def query_graph(storage: BaseGraphStorage, query: str, model: str) -> any:
    """Pipeline to retrieve context and generate an answer."""
    retriever = GraphRetriever(storage)
    generator = ResponseGenerator(model=model)
    query_gen = QueryGenerator()
    
    try:
        # Use QueryGenerator to find target entities instead of naive split
        # Since we are using MemoryGraphStorage for this CLI, we extract the target node
        # from the generated Cypher query's parameters.
        cypher, params = query_gen.generate_query(query)
        
        # We try to find the most likely target node from the parameters
        target_node = None
        for val in params.values():
            if isinstance(val, str):
                target_node = val
                break
        
        if not target_node:
            # Fallback to naive split if QueryGenerator fails to provide a parameter
            target_node = query.split()[0]
            
        logger.info(f"Retrieving context for target node: {target_node}")
        context = retriever.retrieve_k_hop_neighborhood(target_node, k=2)
    except Exception as e:
        logger.warning(f"Query generation failed: {e}. Falling back to naive extraction.")
        target_node = query.split()[0]
        retriever = GraphRetriever(storage)
        context = retriever.retrieve_k_hop_neighborhood(target_node, k=2)
    
    response = generator.generate(query, [context])
    return response

def main():
    parser = argparse.ArgumentParser(description="RAG Knowledge Graph Builder CLI")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Build command
    build_parser = subparsers.add_parser("build", help="Build a KG from a file")
    build_parser.add_argument("--file", required=True, help="Path to the source document")
    build_parser.add_argument("--model", default="sciphi/triplex:latest", help="Model for extraction")
    build_parser.add_argument("--save", help="Path to save the resulting graph JSON")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query a KG")
    query_parser.add_argument("--query", required=True, help="The question to ask")
    query_parser.add_argument("--model", default="gemma2", help="Model for generation")
    query_parser.add_argument("--load", help="Path to load a graph JSON file")
    query_parser.add_argument("--file", help="Path to the file to build the graph from (if --load is not used)")

    args = parser.parse_args()

    if args.command == "build":
        try:
            storage = build_graph(args.file, args.model)
            if args.save:
                # We know it's MemoryGraphStorage for now, but we cast for type checking
                if hasattr(storage, 'export_json'):
                    storage.export_json(args.save)
                    logger.info(f"Graph saved to {args.save}")
            print("\nSuccess: Knowledge Graph built.")
        except Exception as e:
            logger.error(f"Build failed: {e}")
            sys.exit(1)

    elif args.command == "query":
        try:
            storage = MemoryGraphStorage()
            if args.load:
                logger.info(f"Loading graph from {args.load}...")
                storage.import_json(args.load)
            elif args.file:
                logger.info(f"Building temporary graph from {args.file}...")
                storage = build_graph(args.file, "sciphi/triplex:latest")
            else:
                logger.error("Either --load or --file must be provided for query.")
                sys.exit(1)
                
            response = query_graph(storage, args.query, args.model)
            print("\n--- Response ---")
            print(response.answer)
            print("\n--- Citations ---")
            for s, p, o in response.citations:
                print(f"{s} --({p})--> {o}")
        except Exception as e:
            logger.error(f"Query failed: {e}")
            sys.exit(1)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
