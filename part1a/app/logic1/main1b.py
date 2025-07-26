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
from main import EnhancedPDFProcessor,OUTPUT_DIR

# Download required NLTK data (do this once)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

from main import EnhancedPDFProcessor, OUTPUT_DIR

INPUT_DIR = "../../input2"


class LightweightEmbedder:
    """Lightweight text embedder using TF-IDF with domain-specific weighting"""
    
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )
        self.stemmer = PorterStemmer()
        self.fitted = False
        
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.\,\!\?]', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text.lower()
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit vectorizer and transform texts"""
        preprocessed = [self.preprocess_text(text) for text in texts]
        embeddings = self.vectorizer.fit_transform(preprocessed)
        self.fitted = True
        return embeddings.toarray()
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts using fitted vectorizer"""
        if not self.fitted:
            raise ValueError("Vectorizer not fitted yet")
        preprocessed = [self.preprocess_text(text) for text in texts]
        embeddings = self.vectorizer.transform(preprocessed)
        return embeddings.toarray()


class SmartRelevanceCalculator:
    """Advanced relevance calculation using multiple signals"""
    
    def __init__(self):
        self.discovered_domains = {}
        self.domain_cache = {}
        self.stemmer = PorterStemmer()
        
    def discover_domains_from_corpus(self, all_texts: List[str], persona: str, task: str) -> Dict[str, List[str]]:
        """Dynamically discover domain-specific keywords from the document corpus"""
        if not all_texts:
            return {}
            
        # Combine all text for analysis
        corpus_text = " ".join(all_texts).lower()
        persona_task_text = f"{persona} {task}".lower()
        
        # Extract domain indicators from persona and task
        domain_indicators = self._extract_domain_indicators(persona_task_text)
        
        # Use TF-IDF to find important terms in the corpus
        vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,  # Must appear in at least 2 documents
            max_df=0.8  # Don't include terms that appear in >80% of documents
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform([corpus_text])
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]
            
            # Get top terms
            top_indices = tfidf_scores.argsort()[-50:][::-1]  # Top 50 terms
            important_terms = [feature_names[i] for i in top_indices if tfidf_scores[i] > 0]
            
            # Group terms by semantic similarity/context
            discovered_domains = self._cluster_terms_by_context(important_terms, persona_task_text)
            
            return discovered_domains
            
        except Exception as e:
            print(f"Warning: Domain discovery failed: {e}")
            # Fallback to simple frequency analysis
            return self._fallback_domain_discovery(corpus_text, persona_task_text)
    
    def _extract_domain_indicators(self, text: str) -> List[str]:
        """Extract potential domain indicators from persona/task description"""
        # Common domain-indicating words
        domain_patterns = {
            'academic': r'\b(?:research|study|academic|scholar|university|paper|thesis|dissertation)\b',
            'business': r'\b(?:business|company|corporate|market|sales|revenue|profit|strategy)\b',
            'technical': r'\b(?:software|system|technical|engineering|development|programming|code)\b',
            'medical': r'\b(?:medical|health|patient|clinical|treatment|diagnosis|therapy)\b',
            'legal': r'\b(?:legal|law|court|attorney|contract|regulation|compliance)\b',
            'financial': r'\b(?:financial|finance|investment|banking|accounting|budget|economic)\b',
            'educational': r'\b(?:teaching|learning|education|student|curriculum|pedagogy)\b',
            'scientific': r'\b(?:experiment|hypothesis|methodology|analysis|data|scientific)\b'
        }
        
        indicators = []
        for domain, pattern in domain_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                indicators.append(domain)
        
        return indicators
    
    def _cluster_terms_by_context(self, terms: List[str], context: str) -> Dict[str, List[str]]:
        """Group terms into contextual clusters"""
        clusters = {'primary': [], 'secondary': [], 'entities': []}
        
        context_words = set(context.split())
        
        for term in terms:
            term_words = set(term.split())
            
            # Check if term appears in context (high relevance)
            if term_words & context_words:
                clusters['primary'].append(term)
            # Check if it's likely an entity (proper noun pattern)
            elif re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$', term):
                clusters['entities'].append(term)
            else:
                clusters['secondary'].append(term)
        
        # Remove empty clusters
        return {k: v for k, v in clusters.items() if v}
    
    def _fallback_domain_discovery(self, corpus_text: str, context: str) -> Dict[str, List[str]]:
        """Fallback method using simple frequency analysis"""
        # Extract frequent meaningful words
        words = re.findall(r'\b[a-z]{4,}\b', corpus_text.lower())
        word_freq = Counter(words)
        
        # Filter out very common words and get top terms
        common_words = {
            'that', 'this', 'with', 'from', 'they', 'have', 'will', 'been', 'were', 
            'said', 'each', 'which', 'their', 'time', 'would', 'there', 'could', 
            'other', 'more', 'very', 'what', 'know', 'just', 'first', 'also', 
            'after', 'back', 'good', 'work', 'well', 'way', 'even', 'new', 'want', 
            'because', 'any', 'these', 'give', 'day', 'us', 'most', 'her', 'world', 
            'over', 'think', 'also', 'its', 'only', 'see', 'him', 'two', 'how'
        }
        
        filtered_terms = [
            word for word, freq in word_freq.most_common(30) 
            if word not in common_words and freq > 1
        ]
        
        return {'discovered': filtered_terms}
        
    def extract_named_entities(self, text: str) -> List[str]:
        """Simple named entity extraction using regex patterns"""
        entities = []
        
        # Common patterns for different entity types
        patterns = {
            'organizations': r'\b[A-Z][a-z]+ (?:Inc|Corp|LLC|Ltd|Company|Organization|Institute|University)\b',
            'dates': r'\b(?:\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|\d{4})\b',
            'numbers': r'\b\d+(?:\.\d+)?%?\b',
            'proper_nouns': r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        }
        
        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, text)
            entities.extend(matches)
            
        return list(set(entities))
    
    def calculate_position_weight(self, section_index: int, total_sections: int) -> float:
        """Calculate weight based on section position (intro/conclusion get higher weights)"""
        if total_sections <= 1:
            return 1.0
            
        # Higher weight for first and last sections
        if section_index == 0 or section_index == total_sections - 1:
            return 1.2
        elif section_index <= 2 or section_index >= total_sections - 3:
            return 1.1
        else:
            return 1.0
    
    def detect_domain_dynamic(self, text: str, discovered_domains: Dict[str, List[str]]) -> Tuple[str, float]:
        """Dynamically detect domain using discovered keywords"""
        if not discovered_domains:
            return 'general', 0.0
            
        text_lower = text.lower()
        domain_scores = {}
        
        for domain, keywords in discovered_domains.items():
            score = 0
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    # Weight longer phrases higher
                    weight = len(keyword.split())
                    score += weight
            
            if keywords:  # Avoid division by zero
                domain_scores[domain] = score / len(keywords)
        
        if domain_scores:
            best_domain = max(domain_scores.items(), key=lambda x: x[1])
            return best_domain[0], best_domain[1]
        else:
            return 'general', 0.0
    
    def calculate_semantic_relevance(self, text: str, query_embedding: np.ndarray, 
                                   embedder: LightweightEmbedder) -> float:
        """Calculate semantic similarity using embeddings"""
        try:
            text_embedding = embedder.transform([text])
            similarity = cosine_similarity(query_embedding.reshape(1, -1), text_embedding)[0][0]
            return max(0.0, similarity)
        except Exception as e:
            print(f"Error calculating semantic relevance: {e}")
            return 0.0


class PersonaPDFProcessor(EnhancedPDFProcessor):
    def __init__(self):
        super().__init__()
        self.keyword_cache = {}
        self.embedder = LightweightEmbedder(max_features=3000)  # Reduced for speed
        self.relevance_calculator = SmartRelevanceCalculator()
        self.stop_words = set(stopwords.words('english'))
        
    def extract_advanced_keywords(self, text: str, top_k: int = 20) -> List[str]:
        """Extract keywords using TF-IDF and frequency analysis"""
        if text in self.keyword_cache:
            return self.keyword_cache[text]
        
        # Clean text
        cleaned_text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = word_tokenize(cleaned_text)
        
        # Filter words
        filtered_words = [
            word for word in words 
            if len(word) > 3 and word not in self.stop_words and word.isalpha()
        ]
        
        # Get word frequencies
        word_freq = Counter(filtered_words)
        
        # Extract phrases (bigrams)
        bigrams = [f"{filtered_words[i]} {filtered_words[i+1]}" 
                  for i in range(len(filtered_words)-1)]
        bigram_freq = Counter(bigrams)
        
        # Combine single words and phrases
        keywords = []
        keywords.extend([word for word, _ in word_freq.most_common(top_k//2)])
        keywords.extend([phrase for phrase, _ in bigram_freq.most_common(top_k//2)])
        
        self.keyword_cache[text] = keywords[:top_k]
        return keywords[:top_k]

    def calculate_enhanced_relevance(self, text: str, persona: str, task: str, 
                                   section_index: int, total_sections: int,
                                   discovered_domains: Dict[str, List[str]] = None) -> float:
        """Calculate relevance using multiple factors with dynamic domain detection"""
        
        # 1. Keyword-based relevance
        text_keywords = set(self.extract_advanced_keywords(text, 15))
        persona_keywords = set(self.extract_advanced_keywords(persona, 10))
        task_keywords = set(self.extract_advanced_keywords(task, 10))
        
        all_query_keywords = persona_keywords.union(task_keywords)
        
        if all_query_keywords:
            keyword_overlap = len(text_keywords.intersection(all_query_keywords))
            keyword_score = min(keyword_overlap / len(all_query_keywords), 1.0)
        else:
            keyword_score = 0.0
        
        # 2. Position-based weight
        position_weight = self.relevance_calculator.calculate_position_weight(
            section_index, total_sections
        )
        
        # 3. Named entities overlap
        text_entities = set(self.relevance_calculator.extract_named_entities(text))
        query_entities = set(self.relevance_calculator.extract_named_entities(f"{persona} {task}"))
        
        if query_entities:
            entity_score = len(text_entities.intersection(query_entities)) / len(query_entities)
        else:
            entity_score = 0.0
        
        # 4. Dynamic domain relevance
        domain_bonus = 0.0
        if discovered_domains:
            text_domain, text_domain_score = self.relevance_calculator.detect_domain_dynamic(
                text, discovered_domains
            )
            query_domain, query_domain_score = self.relevance_calculator.detect_domain_dynamic(
                f"{persona} {task}", discovered_domains
            )
            
            if text_domain == query_domain and text_domain != 'general':
                domain_bonus = min(text_domain_score * query_domain_score * 0.2, 0.2)
        
        # 5. Text quality score (length, structure)
        quality_score = min(len(text.split()) / 100, 1.0)  # Prefer substantive content
        
        # 6. Persona-task alignment score
        alignment_score = self._calculate_persona_task_alignment(text, persona, task)
        
        # Combine all scores with dynamic weighting
        final_score = (
            keyword_score * 0.35 +
            entity_score * 0.25 +
            alignment_score * 0.20 +
            quality_score * 0.10 +
            domain_bonus +
            (position_weight - 1.0) * 0.10
        )
        
        return min(max(final_score, 0.0), 1.0)  # Ensure score is between 0 and 1
    
    def _calculate_persona_task_alignment(self, text: str, persona: str, task: str) -> float:
        """Calculate how well text aligns with persona's needs for the specific task"""
        text_lower = text.lower()
        
        # Extract action words from task
        action_patterns = r'\b(?:analyze|study|understand|learn|research|find|identify|compare|evaluate|assess|review|summarize|explain|determine|investigate|examine)\b'
        task_actions = re.findall(action_patterns, task.lower())
        
        # Extract role-specific indicators from persona
        role_patterns = r'\b(?:student|teacher|researcher|analyst|manager|developer|engineer|doctor|lawyer|consultant|salesperson|entrepreneur)\b'
        persona_roles = re.findall(role_patterns, persona.lower())
        
        # Check alignment
        alignment_score = 0.0
        
        # Action alignment
        if task_actions:
            action_score = sum(1 for action in task_actions if action in text_lower) / len(task_actions)
            alignment_score += action_score * 0.6
        
        # Role-specific content alignment
        if persona_roles:
            role_score = sum(1 for role in persona_roles 
                           if any(indicator in text_lower for indicator in self._get_role_indicators(role))) / len(persona_roles)
            alignment_score += role_score * 0.4
        
        return min(alignment_score, 1.0)
    
    def _get_role_indicators(self, role: str) -> List[str]:
        """Get context-specific indicators for different roles"""
        role_indicators = {
            'student': ['assignment', 'homework', 'exam', 'grade', 'course', 'class', 'textbook'],
            'teacher': ['curriculum', 'lesson', 'pedagogy', 'assessment', 'classroom', 'instruction'],
            'researcher': ['methodology', 'hypothesis', 'data', 'experiment', 'publication', 'peer review'],
            'analyst': ['analysis', 'metrics', 'performance', 'trends', 'insights', 'report'],
            'manager': ['strategy', 'team', 'objectives', 'planning', 'leadership', 'decision'],
            'developer': ['code', 'programming', 'software', 'application', 'framework', 'debugging'],
            'engineer': ['design', 'specification', 'implementation', 'testing', 'optimization'],
            'doctor': ['patient', 'diagnosis', 'treatment', 'medical', 'clinical', 'symptoms'],
            'lawyer': ['legal', 'contract', 'regulation', 'compliance', 'court', 'case'],
            'consultant': ['recommendation', 'solution', 'client', 'advisory', 'expertise'],
            'salesperson': ['customer', 'product', 'revenue', 'target', 'conversion', 'relationship'],
            'entrepreneur': ['business', 'startup', 'market', 'innovation', 'funding', 'growth']
        }
        
        return role_indicators.get(role.lower(), [])

    def extract_section_content_smart(self, doc, section_title: str, page_num: int) -> Tuple[str, List[str]]:
        """Extract section content and split into meaningful chunks"""
        content = []
        sentences = []
        found_section = False
        start_page = max(0, page_num - 1)
        
        for page_idx in range(start_page, min(len(doc), page_num + 3)):  # Check nearby pages
            page = doc[page_idx]
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" not in block:
                    continue
                    
                for line in block["lines"]:
                    spans = line.get("spans", [])
                    if not spans:
                        continue
                        
                    text = self.normalize_text(" ".join(span["text"] for span in spans))
                    if not text:
                        continue
                    
                    # Check if we found our section
                    if not found_section:
                        if text.lower().strip() == section_title.lower().strip():
                            found_section = True
                        continue
                    
                    # Stop at next major heading
                    if self.is_likely_heading(text, spans[0]):
                        break
                    
                    content.append(text)
                    
                    # Split into sentences for fine-grained analysis
                    try:
                        text_sentences = sent_tokenize(text)
                        sentences.extend(text_sentences)
                    except Exception as e:
                        print(f"Error tokenizing sentences: {e}")
                        sentences.append(text)
            
            if found_section and content:
                break
        
        return "\n".join(content), sentences
    
    def is_likely_heading(self, text: str, span: dict) -> bool:
        """Simple heading detection"""
        # Check text characteristics
        if len(text.split()) > 15:  # Too long to be a heading
            return False
        
        # Check for heading-like patterns
        heading_patterns = [
            r'^\d+\.?\s+[A-Z]',  # Numbered sections
            r'^[A-Z][A-Z\s]{3,}$',  # ALL CAPS
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*:?$'  # Title Case
        ]
        
        for pattern in heading_patterns:
            if re.match(pattern, text.strip()):
                return True
        
        # Check font properties if available
        if span.get("size", 0) > 12 or span.get("flags", 0) & 16:  # Large or bold
            if len(text.split()) <= 8:
                return True
        
        return False

    def process_collection(self, input_data: Dict, pdf_dir: str) -> Dict:
        """Process collection with enhanced relevance scoring"""
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
        all_sentences = []
        all_texts_for_embedding = []
        corpus_texts = []  # For domain discovery
        
        # First pass: extract all content
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
                
                doc_text_parts = []  # Collect text from this document
                
                for idx, section in enumerate(outline):
                    content, sentences = self.extract_section_content_smart(
                        doc, section["text"], section["page"]
                    )
                    
                    if content and len(content.strip()) > 50:  # Filter very short content
                        full_section_text = f"{section['text']} {content}"
                        doc_text_parts.append(full_section_text)
                        
                        section_data = {
                            "document": filename,
                            "section_title": section["text"],
                            "page_number": section["page"],
                            "content": content,
                            "word_count": len(content.split()),
                            "section_index": idx,
                            "total_sections": len(outline)
                        }
                        
                        all_sections.append(section_data)
                        all_texts_for_embedding.append(full_section_text)
                        
                        # Process sentences for subsection analysis
                        for sentence in sentences:
                            if len(sentence.strip()) > 30:  # Filter short sentences
                                all_sentences.append({
                                    "document": filename,
                                    "text": sentence.strip(),
                                    "page_number": section["page"],
                                    "section_title": section["text"]
                                })
                
                # Add document's text to corpus
                if doc_text_parts:
                    corpus_texts.append(" ".join(doc_text_parts))
                
                doc.close()
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
        
        # Discover domains dynamically from the corpus
        print("Discovering domain-specific keywords...")
        discovered_domains = self.relevance_calculator.discover_domains_from_corpus(
            corpus_texts, persona, task
        )
        
        if discovered_domains:
            print(f"Discovered domains: {list(discovered_domains.keys())}")
            for domain, keywords in discovered_domains.items():
                print(f"  {domain}: {keywords[:5]}...")  # Show first 5 keywords
        
        # Second pass: calculate relevance scores with discovered domains
        print("Calculating relevance scores...")
        for section in all_sections:
            relevance = self.calculate_enhanced_relevance(
                f"{section['section_title']} {section['content']}",
                persona,
                task,
                section["section_index"],
                section["total_sections"],
                discovered_domains
            )
            section["relevance_score"] = relevance
        
        # Calculate sentence relevance scores
        for sentence in all_sentences:
            sentence_relevance = self.calculate_enhanced_relevance(
                sentence["text"], persona, task, 0, 1, discovered_domains
            )
            sentence["relevance_score"] = sentence_relevance
        
        # Third pass: enhance with semantic similarity if we have content
        if all_texts_for_embedding:
            try:
                print("Computing semantic similarities...")
                # Create embeddings for all texts
                text_embeddings = self.embedder.fit_transform(all_texts_for_embedding)
                
                # Create query embedding
                query_text = f"{persona} {task}"
                query_embedding = self.embedder.transform([query_text])[0]
                
                # Update relevance scores with semantic similarity
                for i, section in enumerate(all_sections):
                    semantic_score = self.relevance_calculator.calculate_semantic_relevance(
                        section["content"], query_embedding, self.embedder
                    )
                    
                    # Combine with existing score (weighted more towards content-based scoring)
                    combined_score = (section["relevance_score"] * 0.75 + semantic_score * 0.25)
                    section["relevance_score"] = combined_score
                    
            except Exception as e:
                print(f"Warning: Semantic analysis failed: {e}")
        
        # Sort and select top sections
        all_sections.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Add top sections to results
        for rank, section in enumerate(all_sections[:5], 1):
            results["extracted_sections"].append({
                "document": section["document"],
                "section_title": section["section_title"],
                "importance_rank": rank,
                "page_number": section["page_number"]
            })
            
            print(f"Rank {rank}: {section['section_title']} (score: {section['relevance_score']:.3f})")
        
        # Sort and select top sentences for subsection analysis
        all_sentences.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Deduplicate similar sentences
        selected_sentences = []
        for sentence in all_sentences:
            if len(selected_sentences) >= 5:
                break
                
            is_duplicate = False
            sentence_words = set(sentence["text"].lower().split())
            
            for selected in selected_sentences:
                selected_words = set(selected["text"].lower().split())
                # Check for high overlap (>70% similar)
                overlap = len(sentence_words & selected_words)
                if overlap > len(sentence_words) * 0.7:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                selected_sentences.append(sentence)
        
        for sentence in selected_sentences:
            results["subsection_analysis"].append({
                "document": sentence["document"],
                "refined_text": sentence["text"],
                "page_number": sentence["page_number"]
            })
        
        return results


def main():
    processor = PersonaPDFProcessor()
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