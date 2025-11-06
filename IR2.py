# Simple Document Retrieval using an Inverted Index

# Step 1: Take multiple documents as input from the user
docs = []
print("Enter documents (type 'stop' to finish):")
while True:
    d = input("> ")                    # Read each document line
    if d.lower() == "stop":            # Stop when user types 'stop'
        break
    docs.append(d.lower())             # Convert to lowercase for uniformity

# Step 2: Create an inverted index
# The inverted index maps each word to the list of document numbers it appears in
index = {}
for i, d in enumerate(docs, 1):        # Enumerate documents (start index from 1)
    for w in d.split():                # Split document into words
        index.setdefault(w, []).append(i)  # Add document ID to the word's list

# Step 3: Take search query input from the user
q = input("\nEnter search words: ").lower().split()  # Convert to lowercase and split into words

# Step 4: Initialize result with all document numbers
res = set(range(1, len(docs) + 1))

# Step 5: Perform intersection of documents containing each query word
# This ensures only documents containing *all* the search words are returned
for w in q:
    res &= set(index.get(w, []))       # Intersect with docs containing current word

# Step 6: Display results
print("\nInverted Index:", index)      # Show word-to-document mapping
print("Matched Documents:", list(res) if res else "No match found")  # Final matching docs
