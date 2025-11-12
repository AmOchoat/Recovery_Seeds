from candidate import ChunkCandidate
from basic_enumerator import BasicKeyEnumerator
from bitarray import bitarray

L0 = [
    ChunkCandidate(0.9, bitarray('01')),
    ChunkCandidate(0.6, bitarray('00'))
]

L1 = [
    ChunkCandidate(0.8, bitarray('10')),
    ChunkCandidate(0.5, bitarray('11'))
]

enumerator = BasicKeyEnumerator(L0, L1)

while True:
    cand = enumerator.next_candidate()
    if cand is None:
        break
    print(cand)

