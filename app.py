import gradio as gr
import pypdf
import google.generativeai as genai

GOOGLE_API_KEY = 'AIzaSyD1v_oiuH9m3ybhLhDfOeDWMpb1OkmbSXk'

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

def get_response_from_pdf(pdf_file, question):
    """
    Extracts text from a PDF, and generates a response to a question based on the text.

    Args:
        pdf_file: The uploaded PDF file object.
        question: The user's question about the PDF content.

    Returns:
        A string containing the model's answer.
    """
    if pdf_file is None:
        return "Please upload a PDF document."
    if not question:
        return "Please ask a question."

    try:
        pdf_reader = pypdf.PdfReader(pdf_file.name)
        pdf_text = ""
        for page in pdf_reader.pages:
            pdf_text += page.extract_text()

        if not pdf_text:
            return "Could not extract text from the PDF. The PDF might be image-based or empty."

        prompt = f"""
        Based on the following text from a PDF document, please answer the question.
        
        PDF Text:
        ---
        {pdf_text}
        ---
        
        Question: {question}
        
        Answer:
        """

        # Generate the response
        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"An error occurred: {e}"

# Create the Gradio interface
iface = gr.Interface(
    fn=get_response_from_pdf,
    inputs=[
        gr.File(label="Upload PDF", file_types=[".pdf"]),
        gr.Textbox(label="Question")
    ],
    outputs=gr.Textbox(label="Answer", lines=10),
    title="PDF Q&A with Gemini",
    description="Upload a PDF, ask a question, and get an answer from the document.",
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()
