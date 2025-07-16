# Projectseminar: AI-based Applications – Sommersemester 2025

**Leuphana Universität Lüneburg**  
**Prof:** Dr. Debayan Banerjee, Prof. Dr. Ricardo Usbeck

# AI-Based Web App for Analyzing Sustainability Reports

## 🧾 Project Overview
This project is a web-based application that enables users to upload sustainability reports (PDF format) and extract structured, meaningful insights using a large language model (LLM). Users can interact with the document via a chat interface, compare two reports side-by-side, and download extracted information in JSON format.

Developed as part of our university seminar project (Summer Semester 2025), the application integrates a custom LLM API hosted by the University of Göttingen. It prioritizes performance, usability, and privacy.

---

## ✨ Key Features
- **PDF Upload**: Upload one or two sustainability reports for analysis
- **Chat with Reports**: Ask questions about the content, including follow-up questions with full context
- **LLM-Powered Processing**: Uses the Göttingen-hosted LLM API
- **Chunk-Based Analysis**: Splits PDFs into smaller chunks for improved response accuracy
- **JSON Output**: Extracted key data (e.g., CO₂ emissions, climate actions) downloadable as JSON
- **Chat Export**: Download complete chat transcript
- **Key Insights Export**: One-click download of summarized insights
- **PDF Viewer**: Scrollable viewer for one or both uploaded documents
- **Optimized Frontend**: Simple, clear interface with download and navigation buttons

---

## 🛠 Technologies Used
- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Python (Flask)
- **PDF Parsing**: PyMuPDF (`fitz`)
- **Text Chunking**: LangChain (`CharacterTextSplitter`)
- **LLM API**: University of Göttingen-hosted model
- **Data Format**: JSON

---

## 🚀 How It Works
1. Upload one or two sustainability report PDFs.
2. Ask a question about the content using the chat interface.
3. The server splits the documents into chunks and sends each chunk to the LLM API.
4. The API processes the question and returns chunk-specific responses.
5. Answers are compiled and displayed.
6. Optional: Download the chat transcript and extracted data in JSON format.

---

## 👥 Team
	•	Salih Südan – Presentation & coordination, Backend development, full-stack support
	•	Lennart Schönwald – Backend development, API integration
	•	Piet Storm – Backend development, PDF processing
	•	Kevin Guminski – Frontend development, UI/UX design
	•	All team members contributed to GitHub coordination and documentation

---

## 📄 License

This project was developed for educational purposes. Any commercial or production use must comply with the licensing terms of the University of Göttingen LLM API.

---

## 🧪 Example Use Case
```text
1. Upload your company’s 2023 sustainability report.
2. Ask: "What were the total CO₂ emissions in 2023?"
3. Follow up with: "How does that compare to 2022?"
4. Download the structured insights in JSON format.

