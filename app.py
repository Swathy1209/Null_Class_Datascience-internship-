import streamlit as st
from medical_retriver import MedicalQuestionRetriever

def main():
    st.title("🏥 Medical Q&A Chatbot")

    try:
        retriever = MedicalQuestionRetriever('c:/Users/swathiga/Downloads/MedQuAD_Cleaned.csv')
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return

    user_query = st.text_input("Enter your medical question:")

    if user_query:
        results = retriever.retrieve_answer(user_query)

        if results:
            st.subheader("Top Answers")
            for idx, result in enumerate(results, 1):
                st.markdown(f"**Q:** {result['question']}")
                st.markdown(f"**A:** {result['answer']}")
                st.markdown(f"🔹 Confidence: {result['confidence']}%")
                st.write("---")
        else:
            st.warning("No relevant answers found.")

if __name__ == '__main__':
    main()
