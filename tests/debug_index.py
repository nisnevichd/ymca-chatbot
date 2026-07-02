# debug_index.py
import chromadb

from chatbot import DEFAULT_COLLECTION_NAME, DEFAULT_PERSIST_DIR
client = chromadb.PersistentClient(path=str(DEFAULT_PERSIST_DIR))  # your actual path
collection = client.get_collection(name=DEFAULT_COLLECTION_NAME)
print(collection.count())  # how many chunks total
# add to debug_index.py, replace the limit=5 block
results = collection.get(limit=190)
junk_count = 0
for doc in results['documents']:
    if doc.count('.') > 30 or doc.count('...') > 5:  # crude ToC/dot-leader detector
        junk_count += 1
print(f"Likely junk chunks: {junk_count} / {len(results['documents'])}")