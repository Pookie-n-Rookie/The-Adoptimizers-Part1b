import os
import json
import re
import fitz  # PyMuPDF
from typing import List, Dict, Optional, Tuple
import unicodedata
from datetime import datetime
from collections import defaultdict, Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
import pickle
import gzip
from main import EnhancedPDFProcessor, OUTPUT_DIR

# Download required NLTK data (do this once)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

INPUT_DIR = "../../input2"


class GenericRelevanceCalculator:
    """Generic relevance calculator that works for any persona/task combination"""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = None
        
    def extract_query_keywords(self, persona: str, task: str, top_k: int = 15) -> Dict[str, List[str]]:
        """Extract keywords from persona and task descriptions dynamically"""
        combined_text = f"{persona} {task}".lower()
        
        # Extract meaningful words (filter out stop words and short words)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', combined_text)
        filtered_words = [word for word in words if word not in self.stop_words]
        
        # Get word frequencies
        word_freq = Counter(filtered_words)
        
        # Extract key phrases (bigrams and trigrams)
        word_list = combined_text.split()
        bigrams = [f"{word_list[i]} {word_list[i+1]}" for i in range(len(word_list)-1)]
        trigrams = [f"{word_list[i]} {word_list[i+1]} {word_list[i+2]}" for i in range(len(word_list)-2)]
        
        # Filter meaningful phrases
        meaningful_bigrams = [bg for bg in bigrams if not any(word in self.stop_words for word in bg.split())]
        meaningful_trigrams = [tg for tg in trigrams if not any(word in self.stop_words for word in tg.split())]
        
        # Combine and get top keywords
        all_terms = []
        all_terms.extend([word for word, _ in word_freq.most_common(top_k//2)])
        all_terms.extend(Counter(meaningful_bigrams).most_common(top_k//4))
        all_terms.extend(Counter(meaningful_trigrams).most_common(top_k//4))
        
        # Flatten and get unique terms
        flattened_terms = []
        for term in all_terms:
            if isinstance(term, tuple):
                flattened_terms.append(term[0])
            else:
                flattened_terms.append(term)
        
        # Categorize keywords based on linguistic patterns
        keywords = {
            'action_words': [],
            'domain_concepts': [],
            'descriptors': [],
            'entities': []
        }
        
        # Common action word patterns
        action_patterns = r'\b(?:plan|analyze|study|create|develop|manage|design|implement|research|review|summarize|organize|find|identify|compare|evaluate|assess|examine|investigate|understand|learn|teach|write|build|solve|optimize|improve|measure|track|monitor|report|present|communicate|coordinate|lead|support|guide|advise|recommend|suggest|propose|execute|deliver|achieve|accomplish|complete|finish|process|handle|deal|work|focus|specialize|expert|professional|experienced|skilled|knowledgeable)\b'
        
        for term in flattened_terms:
            term_lower = term.lower()
            
            # Categorize based on patterns
            if re.search(action_patterns, term_lower):
                keywords['action_words'].append(term_lower)
            elif len(term.split()) == 1 and len(term) > 4:  # Single meaningful words
                keywords['domain_concepts'].append(term_lower)
            elif len(term.split()) > 1:  # Multi-word phrases
                keywords['descriptors'].append(term_lower)
            elif term[0].isupper():  # Potential entities
                keywords['entities'].append(term_lower)
        
        # Remove duplicates and empty categories
        for category in keywords:
            keywords[category] = list(set(keywords[category]))
        
        return {k: v for k, v in keywords.items() if v}
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using TF-IDF vectors"""
        try:
            if not hasattr(self, '_temp_vectorizer') or self._temp_vectorizer is None:
                self._temp_vectorizer = TfidfVectorizer(
                    max_features=1000,
                    ngram_range=(1, 2),
                    stop_words='english',
                    lowercase=True,
                    min_df=1
                )
            
            # Combine texts to fit vectorizer
            texts = [text1, text2]
            tfidf_matrix = self._temp_vectorizer.fit_transform(texts)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return max(0.0, similarity)
            
        except Exception as e:
            # Fallback to simple word overlap
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            if not words1 or not words2:
                return 0.0
            
            overlap = len(words1 & words2)
            union = len(words1 | words2)
            return overlap / union if union > 0 else 0.0
    
    def calculate_keyword_relevance(self, text: str, query_keywords: Dict[str, List[str]]) -> float:
        """Calculate relevance based on keyword matching with different weights"""
        text_lower = text.lower()
        
        # Weight different types of keywords differently
        weights = {
            'action_words': 0.35,
            'domain_concepts': 0.30,
            'descriptors': 0.25,
            'entities': 0.10
        }
        
        total_score = 0.0
        
        for category, keywords in query_keywords.items():
            if not keywords:
                continue
                
            # Count matches in this category
            matches = 0
            for keyword in keywords:
                if keyword in text_lower:
                    # Give higher weight to longer, more specific keywords
                    match_weight = len(keyword.split())
                    matches += match_weight
            
            # Normalize by category size and apply weight
            category_score = (matches / len(keywords)) * weights.get(category, 0.1)
            total_score += category_score
        
        return min(total_score, 1.0)
    
    def calculate_content_quality(self, text: str) -> float:
        """Calculate content quality based on various factors"""
        if not text or len(text.strip()) < 20:
            return 0.0
        
        words = text.split()
        sentences = sent_tokenize(text)
        
        # Length factor (optimal around 100-300 words)
        word_count = len(words)
        if word_count < 50:
            length_score = word_count / 50.0
        elif word_count > 300:
            length_score = max(0.7, 1.0 - (word_count - 300) / 1000.0)
        else:
            length_score = 1.0
        
        # Sentence structure score
        if sentences:
            avg_sentence_length = word_count / len(sentences)
            # Prefer moderate sentence length (10-25 words)
            if 10 <= avg_sentence_length <= 25:
                structure_score = 1.0
            else:
                structure_score = max(0.5, 1.0 - abs(avg_sentence_length - 17.5) / 17.5)
        else:
            structure_score = 0.0
        
        # Information density (unique words ratio)
        unique_words = len(set(word.lower() for word in words if word.isalpha()))
        density_score = min(unique_words / word_count if word_count > 0 else 0, 1.0)
        
        # Completeness (well-formed sentences)
        complete_sentences = sum(1 for s in sentences 
                               if len(s.split()) >= 5 and s.strip().endswith(('.', '!', '?')))
        completeness_score = complete_sentences / len(sentences) if sentences else 0.0
        
        # Combine scores
        quality_score = (
            length_score * 0.3 +
            structure_score * 0.25 +
            density_score * 0.25 +
            completeness_score * 0.2
        )
        
        return max(0.0, min(quality_score, 1.0))
    
    def calculate_section_importance(self, section_title: str, section_index: int, total_sections: int) -> float:
        """Calculate section importance based on title and position"""
        importance_score = 0.5  # Base score
        
        # Title-based importance
        title_lower = section_title.lower()
        
        # Important section indicators (generic patterns)
        important_patterns = [
            r'\b(?:introduction|overview|summary|conclusion|main|primary|key|important|essential|core|fundamental|basic|advanced|comprehensive|guide|methodology|approach|strategy|analysis|results|findings|discussion|recommendations|best practices|tips|techniques|methods|principles|concepts|theory|practical|application|implementation|case study|example|demonstration)\b',
            r'\b(?:step|phase|stage|process|procedure|workflow|framework|model|system|structure|architecture|design|development|planning|management|execution|evaluation|assessment|review|audit|optimization|improvement|enhancement|solution|problem|challenge|issue|opportunity|benefit|advantage|feature|capability|functionality|performance|quality|standard|requirement|specification|criteria|objective|goal|target|outcome|result|impact|effect|consequence|implication)\b'
        ]
        
        for pattern in important_patterns:
            if re.search(pattern, title_lower):
                importance_score += 0.3
                break
        
        # Position-based importance
        if total_sections > 1:
            # Slight preference for introduction and conclusion
            if section_index == 0:  # First section (often introduction)
                importance_score += 0.1
            elif section_index == total_sections - 1:  # Last section (often conclusion)
                importance_score += 0.1
            # Middle sections are standard
        
        return min(importance_score, 1.0)


class GenericPersonaPDFProcessor(EnhancedPDFProcessor):
    """Generic PDF processor that works for any persona/task combination"""
    
    def __init__(self):
        super().__init__()
        self.relevance_calculator = GenericRelevanceCalculator()
        self.stop_words = set(stopwords.words('english'))
        
    def extract_section_content_smart(self, doc, section_title: str, page_num: int) -> str:
        """Extract section content with improved text extraction"""
        content_parts = []
        found_section = False
        start_page = max(0, page_num - 1)
        end_page = min(len(doc), page_num + 4)  # Look ahead more pages
        
        for page_idx in range(start_page, end_page):
            page = doc[page_idx]
            blocks = page.get_text("dict")["blocks"]
            
            page_content = []
            
            for block in blocks:
                if "lines" not in block:
                    continue
                
                for line in block["lines"]:
                    spans = line.get("spans", [])
                    if not spans:
                        continue
                    
                    # Extract text from spans
                    line_text = " ".join(span.get("text", "") for span in spans)
                    line_text = self.normalize_text(line_text)
                    
                    if not line_text:
                        continue
                    
                    # Check if this is our target section
                    if not found_section:
                        if self.is_section_match(line_text, section_title):
                            found_section = True
                            continue
                    
                    if found_section:
                        # Stop if we hit another major section
                        if self.is_likely_section_header(line_text, spans[0] if spans else {}):
                            if page_content:  # Save current page content first
                                content_parts.extend(page_content)
                            break
                        
                        # Collect meaningful content
                        if len(line_text.strip()) > 5:  # Filter very short lines
                            page_content.append(line_text)
            
            # Add page content if we found some
            if page_content:
                content_parts.extend(page_content)
            
            # Stop if we found content and hit a section boundary
            if found_section and content_parts and page_content != content_parts:
                break
        
        return "\n".join(content_parts)
    
    def is_section_match(self, text: str, section_title: str) -> bool:
        """Check if text matches section title with fuzzy matching"""
        # Normalize both texts
        text_norm = re.sub(r'[^\w\s]', '', text.lower().strip())
        title_norm = re.sub(r'[^\w\s]', '', section_title.lower().strip())
        
        # Exact match
        if text_norm == title_norm:
            return True
        
        # Fuzzy matching for longer titles
        if len(title_norm.split()) > 1:
            title_words = set(title_norm.split())
            text_words = set(text_norm.split())
            
            # Check word overlap
            overlap = len(title_words & text_words)
            overlap_ratio = overlap / len(title_words)
            
            if overlap_ratio >= 0.7:  # 70% of title words must match
                return True
        
        return False
    
    def is_likely_section_header(self, text: str, span: dict) -> bool:
        """Detect if text is likely a section header"""
        # Skip very long text (unlikely to be headers)
        if len(text.split()) > 15:
            return False
        
        # Check for common header patterns
        header_patterns = [
            r'^\d+\.?\s*[A-Z]',  # Numbered sections (1. Introduction)
            r'^[A-Z][A-Z\s]{4,}$',  # ALL CAPS headers
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*:?$',  # Title Case headers
            r'^[A-Z][a-z]+(?:\s+[a-z]+)*(?:\s+[A-Z][a-z]+)+$'  # Mixed case titles
        ]
        
        for pattern in header_patterns:
            if re.match(pattern, text.strip()):
                return True
        
        # Check font characteristics if available
        font_size = span.get("size", 0)
        font_flags = span.get("flags", 0)
        
        # Large font or bold text with reasonable length
        if (font_size > 12 or (font_flags & 16)) and len(text.split()) <= 10:
            return True
        
        return False
    
    def calculate_comprehensive_relevance(self, section_title: str, content: str, 
                                        persona: str, task: str, section_index: int, 
                                        total_sections: int) -> float:
        """Calculate comprehensive relevance score using multiple factors"""
        
        # Extract query keywords dynamically
        query_keywords = self.relevance_calculator.extract_query_keywords(persona, task)
        
        full_text = f"{section_title} {content}"
        
        # 1. Keyword-based relevance (35%)
        keyword_score = self.relevance_calculator.calculate_keyword_relevance(full_text, query_keywords)
        
        # 2. Semantic similarity with query (25%)
        query_text = f"{persona} {task}"
        semantic_score = self.relevance_calculator.calculate_semantic_similarity(full_text, query_text)
        
        # 3. Content quality (20%)
        quality_score = self.relevance_calculator.calculate_content_quality(content)
        
        # 4. Section importance (15%)
        importance_score = self.relevance_calculator.calculate_section_importance(
            section_title, section_index, total_sections
        )
        
        # 5. Title-specific relevance (5%)
        title_relevance = self.relevance_calculator.calculate_keyword_relevance(section_title, query_keywords)
        
        # Combine all scores
        final_score = (
            keyword_score * 0.35 +
            semantic_score * 0.25 +
            quality_score * 0.20 +
            importance_score * 0.15 +
            title_relevance * 0.05
        )
        
        return max(0.0, min(final_score, 1.0))
    
    def extract_quality_subsections(self, content: str, min_length: int = 80) -> List[str]:
        """Extract high-quality subsections from content"""
        if not content:
            return []
        
        # Split into meaningful chunks
        paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
        
        if not paragraphs:
            return []
        
        # Group paragraphs into meaningful sections
        sections = []
        current_section = []
        current_length = 0
        
        for paragraph in paragraphs:
            paragraph_words = len(paragraph.split())
            
            # Add to current section
            current_section.append(paragraph)
            current_length += paragraph_words
            
            # If section is substantial enough, save it
            if current_length >= min_length:
                sections.append(" ".join(current_section))
                current_section = []
                current_length = 0
        
        # Add remaining content if substantial
        if current_section and current_length >= min_length // 2:
            sections.append(" ".join(current_section))
        
        # Filter and clean sections
        quality_sections = []
        for section in sections:
            # Ensure minimum quality
            if (len(section.split()) >= min_length // 3 and 
                len(section.strip()) > 50 and
                not section.strip().endswith(',')):
                quality_sections.append(section.strip())
        
        return quality_sections[:8]  # Limit to top 8
    
    def process_collection(self, input_data: Dict, pdf_dir: str) -> Dict:
        """Process document collection with generic approach"""
        persona = input_data["persona"]["role"]
        task = input_data["job_to_be_done"]["task"]
        
        print(f"Processing for persona: {persona}")
        print(f"Task: {task}")
        
        results = {
            "metadata": {
                "input_documents": [doc["filename"] for doc in input_data["documents"]],
                "persona": persona,
                "job_to_be_done": task,
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": [],
            "subsection_analysis": []
        }
        
        all_sections = []
        all_subsections = []
        
        # Process each document
        for doc_info in input_data["documents"]:
            filename = doc_info["filename"]
            doc_path = os.path.join(pdf_dir, filename)
            
            try:
                print(f"Processing {filename}...")
                doc = fitz.open(doc_path)
                self._flyer_mode = self.detect_flyer_mode(doc)
                title, exclude = self.extract_title(doc)
                outline = self.extract_outline(doc, exclude)
                
                print(f"Found {len(outline)} sections in {filename}")
                
                for idx, section in enumerate(outline):
                    # Extract section content
                    content = self.extract_section_content_smart(
                        doc, section["text"], section["page"]
                    )
                    
                    # Skip sections with insufficient content
                    if not content or len(content.strip()) < 40:
                        continue
                    
                    # Calculate relevance score
                    relevance_score = self.calculate_comprehensive_relevance(
                        section["text"], content, persona, task, idx, len(outline)
                    )
                    
                    section_data = {
                        "document": filename,
                        "section_title": section["text"],
                        "page_number": section["page"],
                        "content": content,
                        "relevance_score": relevance_score,
                        "word_count": len(content.split())
                    }
                    
                    all_sections.append(section_data)
                    
                    # Extract quality subsections
                    subsection_parts = self.extract_quality_subsections(content)
                    for part in subsection_parts:
                        # Calculate subsection relevance
                        subsection_relevance = self.relevance_calculator.calculate_keyword_relevance(
                            part, self.relevance_calculator.extract_query_keywords(persona, task)
                        )
                        
                        all_subsections.append({
                            "document": filename,
                            "text": part,
                            "page_number": section["page"],
                            "section_title": section["text"],
                            "relevance_score": subsection_relevance
                        })
                
                doc.close()
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
        
        # Sort and select top sections
        all_sections.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Ensure diversity across documents
        selected_sections = []
        used_documents = Counter()
        
        # First pass: select highest scoring sections with document diversity
        for section in all_sections:
            if len(selected_sections) >= 5:
                break
            
            # Prefer documents that haven't been selected yet
            if used_documents[section["document"]] < 2 or len(selected_sections) < 3:
                selected_sections.append(section)
                used_documents[section["document"]] += 1
        
        # Fill remaining slots if needed
        remaining_sections = [s for s in all_sections if s not in selected_sections]
        for section in remaining_sections:
            if len(selected_sections) >= 5:
                break
            selected_sections.append(section)
        
        # Add to results
        for rank, section in enumerate(selected_sections[:5], 1):
            results["extracted_sections"].append({
                "document": section["document"],
                "section_title": section["section_title"],
                "importance_rank": rank,
                "page_number": section["page_number"]
            })
            
            print(f"Rank {rank}: {section['section_title']} "
                  f"(score: {section['relevance_score']:.3f}, doc: {section['document']})")
        
        # Process subsections
        all_subsections.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Select diverse, high-quality subsections
        selected_subsections = []
        used_content_signatures = set()
        
        for subsection in all_subsections:
            if len(selected_subsections) >= 5:
                break
            
            # Create content signature to avoid duplicates
            words = subsection["text"].split()
            signature = " ".join(words[:8] + words[-3:])  # First 8 + last 3 words
            
            if signature not in used_content_signatures and len(words) >= 25:
                selected_subsections.append(subsection)
                used_content_signatures.add(signature)
        
        for subsection in selected_subsections:
            results["subsection_analysis"].append({
                "document": subsection["document"],
                "refined_text": subsection["text"],
                "page_number": subsection["page_number"]
            })
        
        return results


def main():
    processor = GenericPersonaPDFProcessor()
    base_input_dir = INPUT_DIR
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process each collection directory
    for collection_name in os.listdir(base_input_dir):
        collection_dir = os.path.join(base_input_dir, collection_name)
        
        if os.path.isdir(collection_dir):
            print(f"\n{'='*50}")
            print(f"Processing collection: {collection_name}")
            print(f"{'='*50}")
            
            # Find input JSON and PDF directory
            input_json = None
            pdf_dir = os.path.join(collection_dir, "PDFs")
            
            for f in os.listdir(collection_dir):
                if f.endswith(".json"):
                    input_json = os.path.join(collection_dir, f)
                    break
            
            if input_json and os.path.exists(pdf_dir):
                try:
                    with open(input_json, "r", encoding="utf-8") as f:
                        input_data = json.load(f)
                    
                    start_time = datetime.now()
                    result = processor.process_collection(input_data, pdf_dir)
                    end_time = datetime.now()
                    
                    processing_time = (end_time - start_time).total_seconds()
                    print(f"Processing completed in {processing_time:.2f} seconds")
                    
                    # Save output
                    output_filename = f"output_{collection_name}.json"
                    output_path = os.path.join(OUTPUT_DIR, output_filename)
                    
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                    print(f"✓ Successfully processed {collection_name} → {output_filename}")
                    
                except Exception as e:
                    print(f"✗ Failed to process {collection_name}: {str(e)}")
                    import traceback
                    traceback.print_exc()
            else:
                missing = []
                if not os.path.exists(pdf_dir):
                    missing.append("PDFs directory")
                if not input_json:
                    missing.append("JSON input file")
                print(f"⚠ Skipping {collection_name} - missing {' and '.join(missing)}")


if __name__ == "__main__":
    main()