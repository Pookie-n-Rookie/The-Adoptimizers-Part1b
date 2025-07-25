import os
import json
import re
import fitz  # PyMuPDF
from typing import List, Dict, Optional, Tuple
from collections import Counter

INPUT_DIR = "./input"
OUTPUT_DIR = "./output"

class HybridPDFProcessor:
    def __init__(self):
        self.heading_config = {
            'H1': {'styles': ['bold'], 'pattern': r'^[A-Z][A-Za-z0-9 \-\:\,\']+$'},
            'H2': {'styles': ['bold'], 'pattern': r'^[A-Z][A-Za-z0-9 \-\:\,\']+$'},
            'H3': {'styles': ['bold'], 'pattern': r'^[A-Za-z0-9 \-\:\,\']+$'},
            'H4': {'styles': ['bold'], 'pattern': r'^[A-Za-z0-9 \-\:\,\']+$'},
        }
        self.title_pattern = r'^[A-Z][A-Za-z0-9 \-\:\,\']{10,}'
        self.font_thresholds = {}
        self.flyer_mode = False

    def calibrate_font_sizes(self, doc) -> Dict[str, float]:
        font_sizes = []
        for page in doc:
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        if span.get("text", "").strip():
                            font_sizes.append(round(span["size"], 1))

        if not font_sizes:
            return {'H1': 18, 'H2': 14, 'H3': 12, 'H4': 10}

        size_counts = Counter(font_sizes).most_common()
        sorted_sizes = sorted(set(size for size, _ in size_counts), reverse=True)

        return {
            'H1': sorted_sizes[0] if len(sorted_sizes) > 0 else 18,
            'H2': sorted_sizes[1] if len(sorted_sizes) > 1 else 14,
            'H3': sorted_sizes[2] if len(sorted_sizes) > 2 else 12,
            'H4': sorted_sizes[3] if len(sorted_sizes) > 3 else 10,
        }

    def detect_flyer_mode(self, doc) -> bool:
        if len(doc) > 2:
            return False
        bold_blocks = 0
        total_words = 0
        for page in doc:
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    if not spans:
                        continue
                    text = " ".join(span["text"] for span in spans).strip()
                    total_words += len(text.split())
                    font = spans[0].get("font", "").lower()
                    if 'bold' in font:
                        bold_blocks += 1
        return total_words < 500 and bold_blocks >= 3

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
        return candidates[0][0], candidates[0][0]

    def classify_heading(self, span: Dict, text: str) -> Optional[str]:
        size = span.get("size", 0)
        font = span.get("font", "").lower()
        if not text or len(text.split()) < 2:
            return None

        for level in ['H1', 'H2', 'H3', 'H4']:
            threshold = self.font_thresholds.get(level, 0)
            config = self.heading_config[level]
            if (abs(size - threshold) < 1.5 and
                all(s in font for s in config['styles']) and
                re.match(config['pattern'], text)):
                return level
        return None

    def score_heading(self, span: Dict, text: str, avg_size: float, y: float, page_height: float) -> Tuple[Optional[str], float]:
        size = span.get("size", 0)
        font = span.get("font", "").lower()
        text = text.strip()
        if not text or len(text.split()) > (18 if self.flyer_mode else 15):
            return None, 0

        score = 0
        if size > avg_size * 1.2:
            score += 3
        elif size > avg_size * 1.1:
            score += 2
        if 'bold' in font:
            score += 2
        if 2 <= len(text.split()) <= 10:
            score += 1
        if re.match(r'^[A-Z][A-Za-z0-9 \-\:\,\']+$', text):
            score += 1
        if y > page_height * 0.7 and not self.flyer_mode:
            score -= 2

        if score >= (3 if self.flyer_mode else 4):
            return self.classify_heading(span, text) or 'H3', score
        return None, score

    def analyze_font_stats(self, page) -> float:
        sizes = [span["size"] for block in page.get_text("dict")["blocks"]
                 for line in block.get("lines", [])
                 for span in line.get("spans", []) if span.get("text", "").strip()]
        return sum(sizes) / len(sizes) if sizes else 12

    def extract_outline(self, doc, exclude_text: str) -> List[Dict]:
        outline = []
        seen = set()

        for page_num, page in enumerate(doc, start=1):
            blocks = page.get_text("dict")["blocks"]
            avg_size = self.analyze_font_stats(page)
            height = page.rect.height

            for block in blocks:
                spans = []
                for line in block.get("lines", []):
                    spans.extend(line.get("spans", []))
                if not spans:
                    continue

                text = " ".join(span["text"] for span in spans).strip()
                if not text or text in seen or text == exclude_text:
                    continue

                primary_span = max(spans, key=lambda s: s.get("size", 0))
                y = primary_span.get("bbox", [0, 0, 0, 0])[1]
                level, score = self.score_heading(primary_span, text, avg_size, y, height)

                if level:
                    outline.append({
                        "level": level,
                        "text": text,
                        "page": page_num
                    })
                    seen.add(text)

        return self.clean_outline(outline)

    def clean_title(self, title: str) -> str:
        title = re.sub(r'\s+', ' ', title).strip()
        return title if title else "Untitled"

    def clean_outline(self, outline: List[Dict]) -> List[Dict]:
        seen = set()
        cleaned = []
        for item in outline:
            key = (item['text'], item['page'])
            if key not in seen:
                seen.add(key)
                cleaned.append(item)
        return cleaned

    def process_pdf(self, pdf_path: str) -> Dict:
        doc = fitz.open(pdf_path)
        self.font_thresholds = self.calibrate_font_sizes(doc)
        self.flyer_mode = self.detect_flyer_mode(doc)

        title, exclude = self.extract_title(doc)
        outline = self.extract_outline(doc, exclude)
        doc.close()

        return {
            "title": self.clean_title(title),
            "outline": outline
        }

def main():
    processor = HybridPDFProcessor()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".pdf"):
            print(f"Processing: {filename}")
            result = processor.process_pdf(os.path.join(INPUT_DIR, filename))
            with open(os.path.join(OUTPUT_DIR, filename.replace(".pdf", ".json")), "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"Saved: {filename} → {len(result['outline'])} headings")

if __name__ == "__main__":
    main()
