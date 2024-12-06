import streamlit as st
from transformers import pipeline
from transformers import DistilBertForQuestionAnswering
from transformers import DistilBertTokenizerFast, AutoTokenizer
import torch 

def launch_streamlit_ui(models):
    st.sidebar.title("Model Selection")
    selected_model_name = st.sidebar.selectbox(
        "Choose a pre-trained model:",
        options=[model for model in models]  # List of model names
    )

    if selected_model_name == 'Distilled Bert Trained':
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased')
        model.load_state_dict(torch.load('/data/cmpe258-sp24/010892622/DeepDataMiningLearning/nlp/output/QA/squad/savedmodel.pth', map_location=torch.device('cpu')))
        qa_pipeline = pipeline("question-answering", model=model,tokenizer=tokenizer)
    else:
        qa_pipeline = pipeline("question-answering", model=selected_model_name)

    st.title("Question Answering System")
    st.markdown("Provide a context and ask a question. The model will extract an answer from the context.")

    context = st.text_area("Context", placeholder="Enter the context here...", height=200)
    question = st.text_input("Question", placeholder="Enter your question here...")

    if st.button("Get Answer"):
        if context.strip() and question.strip():
            # Perform inference using the QA pipeline
            result = qa_pipeline(question=question, context=context)
            answer = result["answer"]
            score = result["score"]
            st.subheader("Answer:")
            st.write(answer)
            st.subheader("Score:")
            st.write(score)
        else:
            st.warning("Please provide both a context and a question!")

models = ['distilbert-base-uncased', 'deepset/roberta-base-squad2', 'Distilled Bert Trained']
launch_streamlit_ui(models)
