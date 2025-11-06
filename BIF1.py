# DNA sequence (example)
dna = "ATGCGTATGTTAGCGTGAATGCGATAG"

# Find a motif (example motif: "ATG")
motif = "ATG"
motif_positions = []
for i in range(len(dna)):
    if dna[i:i+len(motif)] == motif:
        motif_positions.append(i)

# GC Content Calculation
gc_count = dna.count("G") + dna.count("C")
gc_content = (gc_count / len(dna)) * 100

# Find Coding Regions (ORFs starting with ATG & ending with TAA/TAG/TGA)
stop_codons = ["TAA", "TAG", "TGA"]
orfs = []

for i in range(len(dna)):
    if dna[i:i+3] == "ATG":  # Start codon
        for j in range(i+3, len(dna), 3):
            if dna[j:j+3] in stop_codons:
                orfs.append(dna[i:j+3])
                break

# Print results
print("DNA Sequence:", dna)
print("Motif:", motif)
print("Motif positions:", motif_positions)
print("GC Content: {:.2f}%".format(gc_content))
print("Coding Regions (ORFs):", orfs)
