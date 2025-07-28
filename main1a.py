## main1a.py
import os
import json
import re
import fitz 
from typing import List, Dict, Optional, Tuple
import unicodedata
from collections import Counter

INPUT_DIR = "./input"
OUTPUT_DIR = "./output"

class EnhancedPDFProcessor:
    def __init__(self):
        self.heading_indicators = {
            'font_keywords': ['bold', 'medium', 'black', 'heavy', 'demi'],
            'structural_patterns': {
                'H1': [r'\d+\.\s+[A-Z]', r'^[A-Z][A-Z\s]{2,}$', r'^Chapter\s+\d+', r'^Part\s+[A-Z\d]+'],
                'H2': [r'\d+\.\d+\s+[A-Z]', r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$'],
                'H3': [r'\d+\.\d+\.\d+\s+', r'^[a-z]+(?:\s+[a-z]+)*:?$']
            }
        }
        self.multilingual_patterns = {
            'japanese': r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]',
            'chinese': r'[\u4E00-\u9FFF]',
            'arabic': r'[\u0600-\u06FF]',
            'cyrillic': r'[\u0400-\u04FF]'
        }
        self._flyer_mode = False

    def normalize_text(self, text: str) -> str:
        return unicodedata.normalize('NFKC', re.sub(r'\s+', ' ', text.strip()))

    def detect_language(self, text: str) -> str:
        for lang, pattern in self.multilingual_patterns.items():
            if re.search(pattern, text):
                return lang
        return 'latin'

    def extract_title_candidates(self, doc) -> List[Tuple[str, float, int]]:
        candidates = []
        first_page = doc[0]
        blocks = first_page.get_text("dict")["blocks"]

        for block_idx, block in enumerate(blocks):
            for line_idx, line in enumerate(block.get("lines", [])):
                spans = line.get("spans", [])
                if not spans:
                    continue
                text = self.normalize_text(" ".join(span["text"] for span in spans))
                if len(text) < 5 or len(text) > 200:
                    continue

                span = spans[0]
                font_size = span.get("size", 0)
                font_name = span.get("font", "").lower()
                bbox = span.get("bbox", [0, 0, 0, 0])
                y_position = bbox[1] if bbox else 0
                page_height = first_page.rect.height

                score = 0
                score += min(font_size / 10, 5)
                score += (page_height - y_position) / page_height * 3
                if any(k in font_name for k in self.heading_indicators['font_keywords']):
                    score += 2
                if 3 <= len(text.split()) <= 15:
                    score += 2
                if any(p in text.lower() for p in ['page', 'figure', 'table', 'abstract']):
                    score -= 2
                if self.detect_language(text) != 'latin':
                    score += 1

                candidates.append((text, score, block_idx + line_idx))

        return sorted(candidates, key=lambda x: x[1], reverse=True)

    def extract_title(self, doc) -> Tuple[str, str]:
        candidates = self.extract_title_candidates(doc)
        if not candidates:
            return "", ""
        best_title = candidates[0][0]
        if len(best_title.split()) < 2:
            for title, _, _ in candidates[1:4]:
                if len(title.split()) >= 2:
                    best_title = title
                    break
        return best_title, best_title

    def detect_flyer_mode(self, doc) -> bool:
        if len(doc) > 2:
            return False
        total_words = 0
        bold_blocks = 0
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    if not spans:
                        continue
                    text = self.normalize_text(" ".join(span["text"] for span in spans))
                    total_words += len(text.split())
                    font = spans[0].get("font", "").lower()
                    if any(k in font for k in self.heading_indicators['font_keywords']):
                        bold_blocks += 1
        return total_words < 500 and bold_blocks >= 3

    def calculate_heading_score(self, span: Dict, text: str, page_stats: Dict, y_position: float, page_height: float) -> Tuple[Optional[str], float]:
        font_size = span.get("size", 0)
        font_name = span.get("font", "").lower()
        text = self.normalize_text(text)
        word_count = len(text.split())

        if word_count > (18 if self._flyer_mode else 15):
            return None, 0

        score = 0
        avg_size = page_stats.get('avg_font_size', 12)

        if font_size > avg_size * 1.2:
            score += 3
        elif font_size > avg_size * 1.1:
            score += 2

        if any(k in font_name for k in self.heading_indicators['font_keywords']):
            score += 2

        best_level = None
        for level, patterns in self.heading_indicators['structural_patterns'].items():
            for pattern in patterns:
                if re.match(pattern, text, re.IGNORECASE) and any(k in font_name for k in self.heading_indicators['font_keywords']):
                    best_level = level
                    score += 3
                    break
            if best_level:
                break

        if 2 <= word_count <= 10:
            score += 1

        if self.detect_language(text) != 'latin':
            score += 0.5

        if y_position > page_height * 0.7 and not self._flyer_mode:
            score -= 2

        threshold = 3 if self._flyer_mode else 4
        if score < threshold:
            return None, score

        if best_level:
            return best_level, score
        if font_size >= avg_size * 1.5:
            return 'H1', score
        elif font_size >= avg_size * 1.3:
            return 'H2', score
        elif font_size >= avg_size * 1.1:
            return 'H3', score
        return None, score

    def analyze_page_statistics(self, page) -> Dict:
        sizes = [span.get("size", 0)
                 for block in page.get_text("dict")["blocks"]
                 for line in block.get("lines", [])
                 for span in line.get("spans", []) if span.get("size", 0) > 0]
        return {
            'avg_font_size': sum(sizes) / len(sizes),
            'max_font_size': max(sizes),
            'min_font_size': min(sizes)
        } if sizes else {'avg_font_size': 12, 'max_font_size': 12, 'min_font_size': 12}

    def extract_outline(self, doc, exclude_text: str) -> List[Dict]:
        outline, seen = [], set()
        for page_num, page in enumerate(doc):
            if page_num == 0 and not self._flyer_mode:
                continue  # Skip page 0 for normal documents

            stats = self.analyze_page_statistics(page)
            blocks = page.get_text("dict")["blocks"]
            height = page.rect.height

            for block in blocks:
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    if not spans:
                        continue
                    text = self.normalize_text(" ".join(span["text"] for span in spans))
                    if not text or text in seen or text == exclude_text:
                        continue

                    y = spans[0].get("bbox", [0, 0, 0, 0])[1]
                    level, score = self.calculate_heading_score(spans[0], text, stats, y, height)
                    if level:
                        outline.append({"level": level, "text": text, "page": page_num})
                        seen.add(text)

        return self.post_process_outline(outline)

    def post_process_outline(self, outline: List[Dict]) -> List[Dict]:
        result = []
        for item in outline:
            if not any(
                self.text_similarity(item['text'], other['text']) > 0.7
                for other in result
            ):
                result.append(item)
        return [x for x in result if x['level'] in ['H1', 'H2', 'H3', 'H4']]

    def text_similarity(self, a: str, b: str) -> float:
        s1, s2 = set(a.lower().split()), set(b.lower().split())
        return len(s1 & s2) / len(s1 | s2) if s1 and s2 else 0.0

    def process_pdf(self, pdf_path: str) -> Dict:
        try:
            doc = fitz.open(pdf_path)
            self._flyer_mode = self.detect_flyer_mode(doc)
            title, exclude = self.extract_title(doc)
            outline = self.extract_outline(doc, exclude)
            doc.close()
            return {"title": title, "outline": outline}
        except Exception as e:
            print(f"Error: {e}")
            return {"title": "Untitled", "outline": []}

def main():
    processor = EnhancedPDFProcessor()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".pdf"):
            print(f"Processing {filename}")
            result = processor.process_pdf(os.path.join(INPUT_DIR, filename))
            with open(os.path.join(OUTPUT_DIR, filename.replace(".pdf", ".json")), "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"Saved: {filename} → {len(result['outline'])} headings")

if __name__ == "__main__":
    main()