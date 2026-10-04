import os
import sys
import logging
from rag_builder.extraction.ollama_client import OllamaClient
from rag_builder.extraction.triplets import TripletExtractor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def verify_ollama_pipeline():
    """
    Verification script to check Ollama connectivity and basic triplet extraction.
    """
    print("--- Starting Ollama Pipeline Verification ---\n")
    
    # 1. Test Connectivity
    client = OllamaClient()
    print("Checking connectivity to Ollama (http://localhost:11434)...")
    
    # Support model overrides from environment
    test_model = os.environ.get("OLLAMA_MODEL", "gemma4:31b-cloud")
    try:
        response = client.generate(test_model, "Say hello", format="text")
        if response:
            print(f"✅ Connectivity successful. Model '{test_model}' responded: {response[:50].strip()}...")
        else:
            print(f"❌ Connectivity failed: No response from model '{test_model}'.")
            print("Check if Ollama is running and the model is pulled.")
            return False
    except Exception as e:
        print(f"❌ Connectivity failed with error: {e}")
        return False

    # 2. Test Triplet Extraction with robust error handling
    print(f"\nTesting triplet extraction using model '{test_model}'...")
    extractor = TripletExtractor(client=client, model=test_model)
    
    sample_texts = [
        "Elon Musk founded SpaceX.",
        "Paris is the capital of France.",
        "The Great Wall of China is a historic fortification."
    ]
    
    all_passed = True
    for text in sample_texts:
        print(f"Processing: '{text}'")
        try:
            triplets = extractor.extract(text)
            valid_triplets = [t for t in triplets if t.subject and t.predicate and t.object]
            if valid_triplets:
                for t in valid_triplets:
                    print(f"  -> Found: ({t.subject}, {t.predicate}, {t.object})")
            else:
                print("  -> ⚠️ No valid non-empty triplets extracted.")
                all_passed = False
        except Exception as e:
            print(f"  -> ❌ Extraction error for text '{text}': {e}")
            all_passed = False
    
    if all_passed:
        print("\n✅ Triplet extraction verification passed!")
    else:
        print("\n⚠️ Some triplet extractions failed or returned nothing.")
    
    return all_passed

if __name__ == "__main__":
    success = verify_ollama_pipeline()
    if not success:
        sys.exit(1)
    else:
        print("\n--- Verification Complete ---")
        sys.exit(0)
