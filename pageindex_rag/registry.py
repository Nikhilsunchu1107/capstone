import json

from client import REGISTRY_PATH, pi_client


def build_registry() -> dict:
    registry = {
        d["id"]: {"doc_id": d["id"], "filename": d["name"], "description": d["description"]}
        for d in pi_client.list_documents()["documents"]
    }
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"Registry written: {len(registry)} docs, 0 NIM calls")
    return registry


def load_registry() -> dict:
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    build_registry()
