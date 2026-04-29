import sys
import os
from pprint import pprint

# ensure src dir is importable when running from repo root
sys.path.append(os.path.dirname(__file__))

from entity_resolver import EntityResolver
from graph_builder import KGBuilder


def main():
    # Local model directory under src/models by default
    local_models_root = os.path.join(os.path.dirname(__file__), "models")
    default_local_model = os.path.join(local_models_root, "all-MiniLM-L6-v2")
    builder = KGBuilder(local_model_path=default_local_model)
    resolver = EntityResolver(builder)

    try:
        # Upsert a few sample Person entities with stable keys
        samples = [
            ("e_person_1", "Alan Turing", "Person", "alan turing"),
            ("e_person_2", "A. Turing", "Person", "alan turing"),
            ("e_person_3", "Alan Mathison Turing", "Person", "alan mathison turing"),
        ]

        for key, name, label, norm in samples:
            print(f"Upserting {name} ({key})")
            builder.upsert_entity_by_key(key, name, label, norm_name=norm, domain="people")

        print("Computing and storing embeddings for Person nodes (may download model)")
        resolver.ensure_entity_embeddings("Person")

        mention = "Alan Turing"
        print(f"Disambiguating mention: '{mention}'")
        candidates = resolver.disambiguate_by_embedding("Person", mention_text=mention, threshold=0.6, top_k=5)

        if not candidates:
            print("No candidate exceeded the threshold.")
        else:
            print("Candidates:")
            pprint(candidates)

    except Exception as e:
        print("Demo failed:", e)


if __name__ == "__main__":
    main()
