import os
root = "data/CADICA"
assert os.path.isdir(root), "Put CADICA under data/CADICA (from Mendeley)."
needed = ["selectedVideos", "metadata.xlsx"]
for n in needed:
    p = os.path.join(root, n)
    print(f"{n}: {'OK' if os.path.exists(p) else 'MISSING'} -> {p}")