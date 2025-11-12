from okeanode import random_chunk_candidates, initialize

# Generate 11 chunk lists, each with 2 candidates of 8 bits
chunk_lists = [random_chunk_candidates(2, 8) for _ in range(11)]

# Build OKEA tree
root = initialize(chunk_lists, 0, 10)

# Enumerate all 2048 full candidates using next_candidate()
print("Enumerating all 2048 full key candidates:")
count = 0
while True:
    cand = root.next_candidate()
    if cand is None:
        break
    count += 1
    print(f"{count}: weight={cand.to_weight()}, score={cand.score:.4f}, bits={cand.bits.to01()}")

if count < 2048:
    print(f"⚠️ Only {{count}} candidates generated (expected 2048)")
else:
    print("✅ All 2048 candidates generated successfully.")
