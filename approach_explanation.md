# Approach Explanation - Round 1B: Persona-Driven Document Intelligence

## Round 1A Foundation: Intelligent PDF Structure Extraction

Round 1A established the foundational PDF processing capabilities through the `EnhancedPDFProcessor` class. The Round 1A approach centers on adaptive document analysis that automatically detects document types using `detect_flyer_mode()` to differentiate between dense academic papers and sparse promotional materials, adjusting processing parameters accordingly.

The heading detection system employs multi-factor scoring through `calculate_heading_score()`, combining font size analysis relative to document averages, font weight detection using predefined keywords ('bold', 'medium', 'black'), structural pattern matching for numbered sections, and positional analysis considering page layout. Title extraction uses `extract_title_candidates()` which scores potential titles based on font prominence, page position, and content characteristics.

Multi-language support is achieved through `multilingual_patterns` dictionary covering Japanese, Chinese, Arabic, and Cyrillic scripts, with specialized text normalization using Unicode NFKC normalization. The system handles diverse document formats by analyzing page statistics and adapting thresholds dynamically.

## Round 1B Innovation: Building on Structural Intelligence

Round 1B strategically leverages Round 1A's robust PDF processing foundation while adding persona-driven intelligence. Our `GenericPersonaPDFProcessor` extends the Round 1A `EnhancedPDFProcessor`, inheriting its document parsing capabilities and multilingual support, then adding sophisticated relevance analysis tailored to specific persona requirements.

## Dynamic Keyword Intelligence System

Our `GenericRelevanceCalculator` performs real-time linguistic analysis of persona and task descriptions through `extract_query_keywords()`. The system uses regex patterns to identify action words (analyze, research, develop), extracts domain-specific terminology, identifies multi-word descriptive phrases, and categorizes potential entities. This dynamic approach eliminates the need for domain-specific vocabularies, enabling the system to work across academic research, business analysis, educational content, and any other domain.

## Multi-Dimensional Relevance Framework

The core innovation lies in `calculate_comprehensive_relevance()`, which combines five weighted factors:

**Keyword Relevance (35%)** through `calculate_keyword_relevance()` matches content against dynamically extracted keywords, with different weights for action words (0.35), domain concepts (0.30), descriptors (0.25), and entities (0.10).

**Semantic Similarity (25%)** via `calculate_semantic_similarity()` employs TF-IDF vectorization with cosine similarity, using scikit-learn's `TfidfVectorizer` with 1-2 gram features and English stop words removal.

**Content Quality (20%)** through `calculate_content_quality()` assesses optimal text length (50-300 words), sentence structure coherence, information density via unique word ratios, and content completeness based on well-formed sentences.

**Section Importance (15%)** via `calculate_section_importance()` analyzes structural significance using regex patterns for important section indicators and position-based scoring for introductions and conclusions.

**Title-Specific Relevance (5%)** provides focused header analysis for direct task alignment.

## Intelligent Content Extraction

The `extract_section_content_smart()` method extends Round 1A's section boundary detection with advanced content parsing. It uses `is_section_match()` for fuzzy title matching with 70% word overlap requirements, processes multiple pages (up to 4 pages ahead) for complete section content, and employs `is_likely_section_header()` to detect section boundaries using font characteristics and structural patterns.

## Quality-Driven Subsection Analysis

The `extract_quality_subsections()` method segments content into meaningful chunks with minimum length requirements (80 words default), groups related paragraphs intelligently, and applies quality filtering to ensure substantial, complete content reaches the final output.

## Processing Optimization

Document diversity is ensured through `Counter`-based tracking in `used_documents`, preventing over-representation from single sources. The system processes collections iteratively, applying relevance scoring to all sections, then selecting top performers while maintaining document balance. Subsection analysis uses signature-based duplicate detection to avoid redundant content.

The implementation maintains computational efficiency by leveraging lightweight TF-IDF vectors rather than transformer models, ensuring consistent sub-60-second processing while preserving analytical depth across diverse persona-task combinations.