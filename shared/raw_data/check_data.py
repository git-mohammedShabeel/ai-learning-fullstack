import os, json

base = os.path.join(os.path.expanduser("~"), "Desktop", "DL", "ai-learning-data")

#PDFs
pdfs = [f for f in os.listdir(os.path.join(base, "pdfs")) if f.endswith(".pdf")]
print(f"PDFs:              {len(pdfs)} files")

#QA Pairs
qa_dir = os.path.join(base, "qa_pairs")
for fname in os.listdir(qa_dir):
    if fname.endswith(".json"):
        with open(os.path.join(qa_dir, fname),encoding ="utf-8") as f:
            data = json.load(f)
        print(f"QA - {fname:<30} {len(data):>6} pairs")

img_dir = os.path.join(base, "images")
for cat in os.listdir(img_dir):
    cat_path = os.path.join(img_dir, cat)
    if os.path.isdir(cat_path):
        imgs = os.listdir(cat_path)
        print(f"Images/{cat:<20} {len(imgs):>5} files")

hw_dir = os.path.join(base, "handwritten")
hw_files = os.listdir(hw_dir)
print(f"Handwritten:       {len(hw_files)} images")

print("\nDataset ready for Day 3 (OCR pipeline)")