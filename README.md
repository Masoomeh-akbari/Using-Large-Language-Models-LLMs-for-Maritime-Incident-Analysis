# Using Large Language Models (LLMs) for Maritime Incident Analysis

## Introduction

This project leverages Large Language Models (LLMs) to extract structured information from web-scraped data on maritime incidents, demonstrating the transformative potential of AI in solving complex real-world problems. Conducted in collaboration with Mustafa Amin, Nazanin Hasheminejad, and Jialin He, it was part of the **2024 Math to Power Industry (M2PI)**, a prestigious program held from June 4 to 25, 2024, and organized by PIMS. This intensive three-week, full-time training and work-integrated learning initiative bridges industry challenges with advanced mathematical expertise. For more details about the workshop, visit [M2PI](https://m2pi.ca/2024/).

The project focuses on enhancing national security and maritime defense by converting unstructured text into actionable insights. By targeting key entities such as vessel movements, ownership information, and offenses, it underscores the practical applications of LLMs in addressing critical challenges.

The primary objectives of the project are:
1. **Generate Reports:** Use LLMs to extract and format structured data from web articles.
2. **Evaluate Performance:** Assess the effectiveness of various LLMs in identifying and extracting relevant information from noisy datasets.
3. **Optimize Efficiency:** Explore cost-effective and reliable configurations of free and locally-run LLMs for large-scale information extraction.

---

## Repository Contents

This repository contains the following key components:

- **[`eight-for-loops.py`](./eight-for-loops.py):** Automates the process of extracting structured information from web articles while testing multiple configurations. It iterates through combinations of parameters like chunk size, embedding models, and LLM settings, processes articles through a defined pipeline, and saves results in JSON format for evaluation.

- **[`Scoring_LLM.ipynb`](./Scoring_LLM.ipynb):** Evaluates the performance of LLMs by comparing outputs to human-generated reference data using metrics such as cosine similarity, BLEU, and ROUGE. It calculates a final score to measure alignment with reference outputs.

- **[`Score_ratio-LLM-Comparison.ipynb`](./Score_ratio-LLM-Comparison.ipynb):** Calculates the "Score Ratio" (mean divided by standard deviation) for various metrics across LLMs, identifies the best-performing model for specific entities and metrics, and outputs a detailed comparison.

- **[`Comparison_Plot.ipynb`](./Comparison_Plot.ipynb):** Extracts score data for up to four configurations and generates comparative bar charts for various entities, enabling easy visualization of performance differences.

- **[`M2PI_Team7_FinalReport.pdf`](./M2PI_Team7_FinalReport.pdf):** A detailed explanation of the project’s objectives, methodologies, results, and findings. It delves into the technical aspects of the pipeline, evaluation metrics, and overall performance of different LLMs.

---

## Methodology and Scoring

### Methodology

1. **LLM Selection and Testing:**  
   - Free and locally-run models (e.g., Aya, Gemma, Llama 3, Mistral, Phi3, Qwen2) were prioritized for their cost-effectiveness and data privacy advantages.
   - Commercial models were also analyzed for potential future applications.

2. **Pipeline Implementation:**  
   - Documents were chunked to fit within LLM context limits for efficient processing.  
   - Text embedding was performed using models like `nomic-embed-text` and `mxbai-embed-large` to group semantically similar chunks.  
   - Extracted data was validated against human-generated reports for benchmarking.

### Scoring and Comparison of LLMs

To assess the effectiveness of each LLM, their outputs were evaluated using multiple metrics:  

- **Span-Level Metrics:** Precision, Recall, and F1 Score to assess the accuracy and completeness of extracted entity spans.  
- **Semantic and Text Similarity:** Metrics like Intersection over Union (IoU), BLEU Score, Exact Match (EM), and Levenshtein Distance to evaluate textual overlap and alignment.  
- **Advanced Semantic Analysis:** A WebBERT model was used to assess deeper semantic similarities between model-generated and human-extracted information.

Key findings:
- **Phi-3:** Demonstrated the best overall performance, excelling in both general tasks and challenging entity extraction.  
- **Gemma:** Performed best for extracting complex entities such as "Goods Onboard."  
- **Parameter Optimization:** Adjustments to chunk size, overlap, and top_p significantly improved extraction accuracy.



