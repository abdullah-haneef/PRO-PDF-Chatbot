import streamlit as st
from pdf_utils import extract_text_from_pdf
from embedding_utils import chunk_text, create_faiss_index, retrieve_relevant_chunks
from openai_utils import answer_question_with_chat_gpt
from highlight_utils import get_text_coordinates, highlight_text_in_pdf

def main():
    st.title("Enhanced PDF Q&A with GPT")
    
    # Prompt user for OpenAI API Key
    openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")
    if not openai_api_key:
        st.warning("Please enter your OpenAI API key to proceed.")
        st.stop()

    # File uploader for multiple PDFs
    uploaded_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)
    if not uploaded_files:
        st.info("Please upload at least one PDF to proceed.")
        st.stop()

    # Extract text from all PDFs and keep track of their content
    combined_text = ""
    pdf_file_dict = {}
    for pdf_file in uploaded_files:
        text = extract_text_from_pdf(pdf_file)
        combined_text += text
        pdf_file_dict[pdf_file.name] = text

    # Chunking and FAISS Index Creation
    with st.spinner("Processing documents..."):
        chunks = chunk_text(combined_text, chunk_size=100, overlap=20)
        index, embeddings, model_sbert = create_faiss_index(chunks, model_name='all-MiniLM-L6-v2')
    st.success("Documents processed successfully!")

    # Q&A Section
    st.subheader("Ask a Question about Your Documents")
    user_question = st.text_input("Your Question")
    if user_question:
        with st.spinner("Retrieving relevant chunks..."):
            relevant_chunks = retrieve_relevant_chunks(user_question, index, embeddings, chunks, model_sbert, top_k=3)
        combined_context = "\n".join(relevant_chunks)

        # Use GPT for answering the question
        with st.spinner("Generating answer..."):
            try:
                answer = answer_question_with_chat_gpt(
                    user_question,
                    combined_context,
                    openai_api_key=openai_api_key,
                    model="gpt-3.5-turbo"
                )
                st.write("**Answer:**")
                st.write(answer)

                # Highlight relevant sections in the PDF
                highlight_data = []
                for chunk in relevant_chunks:
                    for pdf_name, text in pdf_file_dict.items():
                        if chunk in text:
                            pdf_file = next(file for file in uploaded_files if file.name == pdf_name)
                            coordinates = get_text_coordinates(pdf_file, chunk)
                            highlight_data.extend([(pdf_name, *coord) for coord in coordinates])

                if highlight_data:
                    st.write("Highlighting relevant sections in the PDF...")
                    for pdf_name, page_num, x0, y0, x1, y1 in highlight_data:
                        input_file = next(file for file in uploaded_files if file.name == pdf_name)
                        output_file = f"highlighted_{pdf_name}"
                        highlight_text_in_pdf(input_file, output_file, highlight_data)
                        st.write(f"Highlighted file saved: {output_file}")

            except Exception as e:
                st.error(f"OpenAI API Error: {e}")

if __name__ == "__main__":
    main()
