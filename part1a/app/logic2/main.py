import os
import json
import re
import fitz  # PyMuPDF
from typing import List, Dict, Optional, Tuple
from collections import Counter

INPUT_DIR = "./input"
OUTPUT_DIR = "./output"

class PDFProcessor:
    def __init__(self):
        # Updated with H4
        self.heading_config = {
            'H1': {'styles': ['bold'], 'pattern': r'^[A-Z][A-Za-z0-9 \-\:\,\']+$'},
            'H2': {'styles': ['bold'], 'pattern': r'^[A-Z][A-Za-z0-9 \-\:\,\']+$'},
            'H3': {'styles': ['bold'], 'pattern': r'^[A-Za-z0-9 \-\:\,\']+$'},
            'H4': {'styles': ['bold'], 'pattern': r'^[A-Za-z0-9 \-\:\,\']+$'},
        }
        self.title_pattern = r'^[A-Z][A-Za-z0-9 \-\:\,\']{10,}'
        self.dynamic_font_sizes = {}

    def calibrate_font_sizes(self, doc) -> Dict[str, float]:
        font_sizes = []
        for page in doc:
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        if span.get("text", "").strip():
                            font_sizes.append(round(span["size"], 1))

        if not font_sizes:
            return {'H1': 18.0, 'H2': 14.0, 'H3': 12.0, 'H4': 10.0}

        size_counts = Counter(font_sizes).most_common()
        sorted_sizes = sorted(set(size for size, _ in size_counts), reverse=True)

        return {
            'H1': sorted_sizes[0] if len(sorted_sizes) > 0 else 18.0,
            'H2': sorted_sizes[1] if len(sorted_sizes) > 1 else 14.0,
            'H3': sorted_sizes[2] if len(sorted_sizes) > 2 else 12.0,
            'H4': sorted_sizes[3] if len(sorted_sizes) > 3 else 10.0,
        }

    def extract_title(self, doc) -> Tuple[str, str]:
        first_page = doc[0]
        blocks = first_page.get_text("dict")["blocks"]
        candidates = []

        for block in blocks:
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                text = " ".join(span["text"] for span in spans).strip()
                size = spans[0].get("size", 0)
                font = spans[0].get("font", "").lower()
                if (re.match(self.title_pattern, text) and
                    len(text.split()) >= 3 and
                    'bold' in font):
                    candidates.append((text, size))

        if not candidates:
            return "Untitled", ""

        candidates.sort(key=lambda x: x[1], reverse=True)
        title = candidates[0][0]
        return title, title

    def classify_heading(self, span: Dict) -> Optional[str]:
        text = span.get("text", "").strip()
        size = span.get("size", 0)
        font = span.get("font", "").lower()

        if not text or len(text.split()) < 2:
            return None

        for level in ['H1', 'H2', 'H3', 'H4']:
            expected_size = self.dynamic_font_sizes.get(level, 0)
            config = self.heading_config[level]
            if (
                abs(size - expected_size) < 1.5 and
                all(style in font for style in config['styles']) and
                re.match(config['pattern'], text)
            ):
                return level
        return None

    def extract_outline(self, doc, exclude_text: str) -> List[Dict]:
        outline = []
        seen_text = set()

        for page_num, page in enumerate(doc, start=1):
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                spans = []
                for line in block.get("lines", []):
                    spans.extend(line.get("spans", []))
                if not spans:
                    continue

                text = " ".join(span["text"] for span in spans).strip()
                if (not text or text in seen_text or
                    text == exclude_text or
                    len(text.split()) > 25):
                    continue

                # Use the largest span in the block to classify
                primary_span = max(spans, key=lambda s: s.get("size", 0))
                level = self.classify_heading(primary_span)

                if level:
                    outline.append({
                        "level": level,
                        "text": text,
                        "page": page_num
                    })
                    seen_text.add(text)

        return outline

    def process_pdf(self, pdf_path: str) -> Dict:
        doc = fitz.open(pdf_path)
        self.dynamic_font_sizes = self.calibrate_font_sizes(doc)

        title, exclude = self.extract_title(doc)
        outline = self.extract_outline(doc, exclude)
        doc.close()

        title = self.clean_title(title)
        outline = self.clean_outline(outline)

        return {
            "title": title,
            "outline": outline
        }

    def clean_title(self, title: str) -> str:
        if title == "Untitled":
            return title
        title = re.sub(r'\s+', ' ', title).strip()
        if len(title) > 1 and title[1].islower():
            title = title[0].upper() + title[1:]
        return title

    def clean_outline(self, outline: List[Dict]) -> List[Dict]:
        seen = set()
        cleaned = []
        for item in outline:
            key = (item['text'], item['page'])
            if key not in seen:
                seen.add(key)
                cleaned.append(item)
        return cleaned

def main():
    processor = PDFProcessor()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for filename in os.listdir(INPUT_DIR):
        if not filename.endswith(".pdf"):
            continue

        try:
            pdf_path = os.path.join(INPUT_DIR, filename)
            print(f"Processing: {filename}")
            result = processor.process_pdf(pdf_path)

            output_path = os.path.join(OUTPUT_DIR, filename.replace(".pdf", ".json"))
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            print(f"Saved: {output_path}")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    main()
