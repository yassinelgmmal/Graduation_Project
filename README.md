# PolySumm: Multimodal Scientific Paper Summarization

## Project Overview

### Problem Statement
The exponential growth of scientific literature presents a significant challenge for researchers seeking to efficiently extract key information. Traditional summarization approaches primarily focus on text, often neglecting critical information embedded within figures and tables. This limitation leads to incomplete understanding and inefficient knowledge acquisition, particularly in fields where visual data representations are crucial.

### Objectives
This project aimed to develop a comprehensive multimodal summarization system for scientific papers. The system integrates information from text, figures, and tables to generate more complete and informative summaries. Our specific objectives included:

1.  **Designing a Robust Pipeline:** To extract structured content from scientific PDF documents.
2.  **Implementing Specialized Summarization Models:** For different content types (text, figures, and tables).
3.  **Integrating Components within a RAG Framework:** To leverage Retrieval-Augmented Generation for enhanced summarization.
4.  **Providing an Accessible User Interface:** To enable researchers to easily utilize this technology.

## Demo Video

A short demonstration of the PolySumm system in action is available below:

[//]: # (Placeholder for demo video)
https://github.com/user-attachments/assets/8098c249-1a95-4532-862e-afe58dbcb282

### Methodology
PolySumm employs a sophisticated system architecture that combines advanced OCR capabilities with specialized summarization models within a Retrieval-Augmented Generation (RAG) framework. The core methodology involves:

*   **Multimodal Data Extraction:** Utilizing PaddleOCR to process scientific PDFs, extracting text, identifying figures, and recognizing table structures. The extracted data is then outputted in a structured JSON format.
*   **Data Preprocessing:** The structured JSON data undergoes rigorous preprocessing, including semantic chunking and embedding generation, before being fed into the RAG pipeline.
*   **Distinct Summarization Approaches:** The system employs different summarization techniques tailored to each content type:
    *   **Text Summarization:** Achieved using transformer-based models.
    *   **Figure Description:** Handled by vision-language models.
    *   **Tabular Data Summarization:** Performed by specialized table-to-text models.
*   **User Interface:** A React-based frontend allows users to upload scientific papers and view the generated multimodal summaries.
*   **Comparative Evaluation:** To validate our approach, we implemented and compared two RAG pipelines: a custom system utilizing in-house summarization models and a second system relying on external APIs (specifically, Azure OpenAI).

### Achievements
The project successfully delivered an effective multimodal scientific paper summarization system. Key achievements include:

*   **Implementation of Two RAG Pipelines:** A custom-built system with in-house summarization models and an API-based system via Azure OpenAI.
*   **Superior Performance of Custom System:** The custom system demonstrated higher accuracy, better integration of figures and tables, faster inference times, and lower resource usage. This makes it particularly well-suited for domain-specific tasks requiring deep multimodal understanding.
*   **Validation of Tailored Approach:** The custom RAG pipeline consistently outperformed the API-based system in terms of accuracy, efficiency, and content completeness, validating the benefits of a tailored, modality-aware summarization approach for scientific documents.

## Key Features

*   **Multimodal Summarization:** Summarizes scientific papers by integrating information from text, figures, and tables.
*   **PDF Processing:** Robust pipeline for extracting structured content from PDF documents.
*   **Advanced OCR:** Leverages PaddleOCR for accurate text, figure, and table recognition.
*   **Retrieval-Augmented Generation (RAG):** Enhances summarization quality by retrieving relevant information.
*   **Specialized Models:** Utilizes transformer-based models for text, vision-language models for figures, and table-to-text models for tables.
*   **User-Friendly Interface:** Intuitive React-based frontend for easy paper uploads and summary viewing.
*   **Comparative Analysis:** Provides insights into the performance of custom vs. API-based RAG implementations.

## Technologies Used

*   **Frontend:** React
*   **OCR:** PaddleOCR
*   **Natural Language Processing (NLP):** Transformer-based models (for text summarization), Vision-Language Models (for figure summarization), Table-to-Text Models (for table summarization)
*   **Backend/Framework:** Retrieval-Augmented Generation (RAG)
*   **Optional Integration:** Azure OpenAI

### Detailed Methodology

#### Data Extraction and Preprocessing
Our system leverages **PaddleOCR** for robust multimodal data extraction from scientific PDFs. This involves:
*   **PDF Processing:** Handling various PDF structures and layouts.
*   **Layout Analysis:** Identifying different content blocks such as text paragraphs, figures, and tables.
*   **Text Recognition:** Accurately extracting text from all identified blocks.
*   **Table and Figure Extraction:** Specifically recognizing and extracting structured data from tables and visual information from figures.
The extracted data is then converted into a structured JSON format, which serves as the input for the subsequent processing steps. This structured data undergoes a crucial preprocessing phase, including **semantic chunking** to break down content into meaningful units and **embedding generation** to create vector representations for efficient retrieval within the RAG pipeline.

#### Summarization Modules
PolySumm integrates several specialized summarization modules, each tailored to a specific data modality:
*   **Figure Summarization:** Utilizes **Vision-Language Models (VLMs)** to generate descriptive summaries of figures, capturing key visual information and its relevance to the paper's content.
*   **Table Summarization:** Employs **Table-to-Text Models** to convert structured tabular data into coherent natural language summaries, highlighting important trends, comparisons, or findings within the tables.
*   **Text Summarization:** Relies on **transformer-based models**, such as **PEGASUS**, for generating abstractive summaries of textual content, ensuring conciseness and information retention.
*   **Topic Classification Model:** An integrated model helps in classifying the topic of the paper, which can further refine the summarization process by focusing on domain-specific aspects.

#### RAG System Core
The Retrieval-Augmented Generation (RAG) framework is central to PolySumm's ability to produce comprehensive and contextually relevant summaries. The RAG integration involves:
*   **Retrieval Strategy:** Based on user queries or the context of the document, relevant chunks of multimodal data (text, figure descriptions, table summaries) are retrieved from the processed document embeddings.
*   **Generation Process:** The retrieved information is then fed into a powerful language model, which generates the final summary, ensuring that it is grounded in the original document's content.
*   **Combining Multimodal Summaries:** A sophisticated mechanism combines the summaries generated from different modalities into a single, cohesive, and informative overall summary.

### Comparative Analysis: Custom vs. API-Based RAG Systems
During development, two distinct RAG pipelines were implemented and rigorously evaluated:
1.  **Custom-Built System:** This system utilizes in-house developed or fine-tuned summarization models for each modality. It demonstrated superior performance in terms of accuracy, deeper integration of figures and tables, faster inference times, and lower resource consumption. This approach proved ideal for achieving domain-specific accuracy and detailed multimodal understanding.
2.  **API-Based System (Azure OpenAI):** This system leveraged external APIs, specifically Azure OpenAI, for its summarization capabilities. While offering fluent generation and quick setup, it exhibited limitations in deep multimodal understanding and integration compared to the custom solution.

The comprehensive evaluation confirmed that the custom RAG pipeline significantly outperformed the API-based counterpart in accuracy, efficiency, and content completeness, underscoring the benefits of a tailored, modality-aware approach for scientific paper summarization.

## Future Work

Potential future enhancements for PolySumm include:
*   Exploring more advanced multimodal fusion techniques.
*   Expanding support for additional document formats and content types.
*   Developing interactive summarization features.
*   Further optimization for real-time processing and deployment in various environments.
*   Investigating techniques to improve OCR accuracy, particularly for complex tables, formulas, and diverse layouts, or incorporating methods to handle OCR uncertainty downstream.
*   Extending the system to take multiple related papers as input and produce a coherent, unified summary that highlights cross-paper themes, differences, and aggregated findings—useful for survey articles or literature reviews.
*   Adapting OCR and summarization pipelines to process papers in languages other than English, including cross-lingual summarization.
*   Incorporating user profiles or feedback loops to tailor summaries to individual reading styles, preferred level of technical detail, or specific interests (e.g., methodology vs. results).
*   Setting up an online learning framework where user corrections or validations feed back into model fine-tuning, gradually improving performance over time. 


