from ascintadb.interface import DatabaseInterface

def test_store_and_retrieve() -> None:
    test_collection = "test_collection"
    test_doc = {"name": "integration_test", "value": 42, "version": "1.0"}

    with DatabaseInterface() as db:
        print("INFO:    Connected to:", db.database_info())
        assert db.count_documents(test_collection) == 0

        db.insert_document(test_collection, test_doc)
        print("INFO:    Document inserted")

        results = db.find_documents(test_collection, {"name": test_doc["name"]})
        assert len(results) == 1 and results[0]["value"] == 42
        print("INFO:    Retrieved documents:", results)

        assert db.count_documents(test_collection) == 1
        db.delete_documents(test_collection, {"name": test_doc["name"]})
        assert db.count_documents(test_collection) == 0
        print("PASS:    Test successful")

def main() -> None:
    test_store_and_retrieve()

if __name__ == "__main__":
    main()
