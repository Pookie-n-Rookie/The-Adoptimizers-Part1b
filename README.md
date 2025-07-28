# Adobe Hackathon - Round 1B: Persona-Driven Document Intelligence

## Project Overview

This repository contains the solution for the Adobe India Hackathon Round 1B by Team "The Adoptimizers" from Jadavpur University, Kolkata, comprising of Swarnendu Banerjee (Team Lead), Sarbo Sarcar, and Swastik Bose. The solution builds upon the PDF outline extraction capabilities from Round 1A to provide intelligent, persona-driven document analysis.

**Round 1B Focus**: Analyzes document collections to extract and rank the most relevant sections based on specific persona requirements and job-to-be-done tasks, utilizing the structured outline extraction from Round 1A as foundation.

Round 1A GitHub Repo : https://github.com/Pookie-n-Rookie/The-Adoptimizers-Part1a

## Architecture

```
round1b-repo/
├── Dockerfile              # Docker configuration
├── main1a.py              # Round 1A model (dependency)
├── main1b.py              # Round 1B: Persona-driven analysis
├── README.md              # This file
├── approach_explanation.md # Methodology explanation
└── requirements.txt       # Python dependencies
```

## Key Features

### Leveraging Round 1A Foundation
- **Structured Document Understanding**: Uses Round 1A's `EnhancedPDFProcessor` for reliable PDF parsing and outline extraction
- **Multi-language Support**: Inherits support for Japanese, Chinese, Arabic, and Cyrillic scripts
- **Smart Heading Detection**: Builds on Round 1A's font analysis and structural pattern recognition

### Round 1B Innovations
- **Generic Persona Processing**: Domain-agnostic system that works across any persona/task combination
- **Dynamic Keyword Extraction**: Automatically extracts relevant terms from persona and task descriptions
- **Multi-factor Relevance Scoring**: Combines keyword matching, semantic similarity, content quality, and section importance
- **Intelligent Content Extraction**: Smart section boundary detection with contextual content extraction
- **Quality Subsection Analysis**: Extracts and ranks meaningful subsections based on relevance

## Dependencies

```
PyMuPDF>=1.23.0          # PDF processing (from Round 1A)
nltk>=3.8                # Natural language processing
scikit-learn>=1.3.0      # Semantic similarity computation
numpy>=1.24.0           # Numerical operations
```

## Docker Configuration

### Build Command
```bash
docker build --platform linux/amd64 -t round1b-solution:latest .
```

### Run Command
```bash
docker run --rm -v "$(pwd)/input:/app/input" -v "$(pwd)/output:/app/output" --network none round1b-solution:latest
```

## Input/Output Specification

### Input Structure
```
/app/input/
├── collection1/
│   ├── input.json          # Persona and task definition
│   └── PDFs/
│       ├── document1.pdf
│       ├── document2.pdf
│       └── document3.pdf
└── collection2/
    ├── input.json
    └── PDFs/
        └── ...
```

### Input JSON Format
```json
{
  "documents": [
    {"filename": "document1.pdf"},
    {"filename": "document2.pdf"}
  ],
  "persona": {
    "role": "PhD Researcher in Computational Biology"
  },
  "job_to_be_done": {
    "task": "Prepare comprehensive literature review focusing on methodologies and benchmarks"
  }
}
```

### Output Format
```json
{
  "metadata": {
    "input_documents": ["doc1.pdf", "doc2.pdf"],
    "persona": "PhD Researcher in Computational Biology",
    "job_to_be_done": "Prepare comprehensive literature review...",
    "processing_timestamp": "2025-01-28T10:30:00"
  },
  "extracted_sections": [
    {
      "document": "doc1.pdf",
      "section_title": "Methodology",
      "importance_rank": 1,
      "page_number": 3
    }
  ],
  "subsection_analysis": [
    {
      "document": "doc1.pdf",
      "refined_text": "The methodology employed in this study...",
      "page_number": 3
    }
  ]
}
```

## How Round 1A Integration Works

1. **PDF Structure Extraction**: Uses `EnhancedPDFProcessor` from Round 1A to extract document outlines
2. **Section Content Mapping**: Leverages Round 1A's heading detection to identify section boundaries
3. **Content Extraction**: Builds on Round 1A's text normalization and multilingual support
4. **Enhanced Analysis**: Adds persona-driven relevance scoring on top of Round 1A's structural understanding

## Performance Specifications

- **Processing Time**: ≤ 60 seconds for document collection (3-5 documents)
- **Model Size**: ≤ 1GB total (including Round 1A components)
- **Architecture**: CPU-only execution (AMD64)
- **Network**: No internet access required (offline operation)

## Usage

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Download required NLTK data (done automatically in code)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Run Round 1B processing
python main1b.py
```

### Docker Execution
The container automatically:
1. Processes all collection directories in `/app/input/`
2. Uses Round 1A model for PDF structure extraction
3. Applies persona-driven analysis for content ranking
4. Generates output JSON files in `/app/output/`

## Technical Architecture

### Round 1A Foundation (`main1a.py`)
- `EnhancedPDFProcessor`: Core PDF processing with `detect_flyer_mode()`, adaptive font analysis
- `extract_title_candidates()`: Title detection using font size, position, and content analysis
- `calculate_heading_score()`: Multi-factor heading detection with font analysis and structural patterns
- `post_process_outline()`: Duplicate removal using `text_similarity()` function
- Multi-language support via `multilingual_patterns` for Japanese, Chinese, Arabic, Cyrillic

### Round 1B Extension (`main.py`)
- `GenericRelevanceCalculator`: Standalone relevance scoring with `extract_query_keywords()` and `calculate_semantic_similarity()`
- `GenericPersonaPDFProcessor`: Extends Round 1A with `extract_section_content_smart()` and `calculate_comprehensive_relevance()`
- `extract_quality_subsections()`: Content segmentation with minimum length requirements
- Document collection processing with diversity optimization using `Counter` for document balance

## Sample Output

<img width="1481" height="822" alt="Image" src="https://github.com/user-attachments/assets/a889140a-466d-4f47-8a2e-eb1c07cd993e" />

## Key Algorithms

1. **Dynamic Keyword Categorization**: Classifies extracted terms into action words, domain concepts, descriptors, and entities using linguistic pattern matching
2. **Multi-dimensional Relevance Scoring**: Weighted combination of keyword relevance (35%), semantic similarity (25%), content quality (20%), section importance (15%), and title relevance (5%)
3. **Smart Content Boundary Detection**: Uses `is_section_match()` with fuzzy text matching and `extract_section_content_smart()` for content extraction across multiple pages
4. **Document Diversity Optimization**: Counter-based document selection ensuring balanced representation with `used_documents` tracking

## Error Handling & Robustness

- Graceful handling of PDF parsing errors from Round 1A
- Fallback mechanisms for content extraction failures
- Comprehensive input validation for persona/task definitions
- Processing time monitoring with early termination safeguards

## Testing Coverage

Successfully tested across diverse scenarios:
- **Academic Research**: Literature reviews, methodology analysis
- **Business Analysis**: Financial reports, market research
- **Educational Content**: Textbook analysis, study guide generation
- **Multi-language Documents**: Various script systems and languages
