from candidate import ChunkCandidate
from basic_enumerator import BasicKeyEnumerator
from bitarray import bitarray

# Define L0 with 5 candidates
L0 = [
    ChunkCandidate(0.95, bitarray('000')),
    ChunkCandidate(0.85, bitarray('001')),
    ChunkCandidate(0.75, bitarray('010')),
    ChunkCandidate(0.65, bitarray('011')),
    ChunkCandidate(0.55, bitarray('100'))
]

# Define L1 with 3 candidates
L1 = [
    ChunkCandidate(0.92, bitarray('110')),
    ChunkCandidate(0.82, bitarray('111')),
    ChunkCandidate(0.72, bitarray('101'))
]

enumerator = BasicKeyEnumerator(L0, L1)

print("Enumerating all 5 × 3 = 15 full key candidates:")
while True:
    cand = enumerator.next_candidate()
    if cand is None:
        break
    print(cand)

