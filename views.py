from django.shortcuts import render,redirect
from rest_framework.parsers import MultiPartParser,FileUploadParser
from  rest_framework.views import APIView
from  rest_framework.response import Response
from .serializers import PDFUploadSerializer,MultiplePDFUploadSerializer
from rest_framework import status
from .models import PDFUpload,FineTunedModel
import pandas as pd
import  json
import glob
import openai
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os
import pdfkit
from io import BytesIO
from django.http import HttpResponse
# from weasyprint import default_url_fetcher, HTML
from langchain_community.document_loaders import UnstructuredExcelLoader
from django.core.files.uploadedfile import InMemoryUploadedFile
from openpyxl import load_workbook, Workbook
from tempfile import NamedTemporaryFile


from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Retrieve the API key
google_api_key = os.getenv("GOOGLE_API_KEY")

# Initialize OpenAI API client
openai.api_key = ''



    
def index(request):
    print("hii")
    return render(request, 'abc.html')






class UploadPDFView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        try:
            excel_file = request.FILES.get('files')
            if not excel_file:
                return Response({"error": "No file uploaded."}, status=status.HTTP_400_BAD_REQUEST)
            
            # Save the uploaded file temporarily using NamedTemporaryFile
            with NamedTemporaryFile(delete=False, suffix='.xlsx') as temp_file:
                for chunk in excel_file.chunks():
                    temp_file.write(chunk)
                temp_file_path = temp_file.name
            
            # Load data from the Excel file
            docs = self.load_excel_data(temp_file_path)
            
            # Clean up the temporary file
            os.remove(temp_file_path)
            self.get_vector_store(docs)
            return Response({"message": "PDF files processed successfully."}, status=status.HTTP_200_OK)
            response = HttpResponse(docs, content_type='application/pdf')
            response['Content-Disposition'] = 'attachment; filename="converted.pdf"'
            return response
        except Exception as e:
            print("inside exception:", e)
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

    def load_excel_data(self, excel_file_path: str):
        loader = UnstructuredExcelLoader(excel_file_path, mode="elements")
        docs = loader.load()
        return docs


    def get_pdf_text(self, pdf_docs):
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

    def get_text_chunks(self, text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
        return chunks

    def get_vector_store(self, text_chunks):
        text_chunks = [doc.page_content for doc in text_chunks]

        # Ensure the texts are not empty
        text_chunks = [text for text in text_chunks if text]
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=os.getenv("GOOGLE_API_KEY"))
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")




import os
import pandas as pd
import uuid

class AnswerQuestionView(APIView):
    def post(self, request, *args, **kwargs):
        try:
            question_file = request.FILES.get('question_file')
            if not question_file:
                return Response({"error": "No file uploaded."}, status=status.HTTP_400_BAD_REQUEST)
            
            questions = self.load_questions_from_excel(question_file)
            if not questions:
                return Response({"error": "No questions found in the uploaded file."}, status=status.HTTP_400_BAD_REQUEST)

            answers = []
            for question in questions:
                response = self.user_input(question)
                answers.append(response["output_text"])

            # Save answers to an Excel file
            # output_path = "C:\\Users\\kunal\\Downloads\\Question_Answer_generate_project\\Question_Answer_generate_project\\media\\uploads\\output_answers.xlsx"
            # self.save_answers_to_excel(questions, answers, output_path)
            responses_dir = os.path.join(settings.MEDIA_ROOT, 'responses')
            os.makedirs(responses_dir, exist_ok=True)
            output_filename = "responses_{}.xlsx".format(uuid.uuid4().hex)
            output_path = os.path.join(settings.MEDIA_ROOT, 'responses', output_filename)
            self.save_answers_to_excel(questions, answers, output_path)

            file_url = request.build_absolute_uri(settings.MEDIA_URL + 'responses/' + output_filename)
            return Response({"message": "Answers generated and saved successfully.", "file_url": file_url}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

    def load_questions_from_excel(self, file):
        df = pd.read_excel(file)
        # Attempt to find the correct column
        possible_column_names = ['Questions', 'questions', 'Question', 'question']
        question_column = None
        for col in possible_column_names:
            if col in df.columns:
                question_column = col
                break

        if question_column is None:
            raise ValueError("No 'Questions' column found in the uploaded file.")
        
        return df[question_column].dropna().tolist()

    def user_input(self, user_question):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = self.get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response

    def get_conversational_chain(self):
        prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in
        provided context, just say, "The answer is not available in the context." Don't provide a wrong answer.
        \n\nContext:\n {context}?\n
        Question: \n{question}\n
        Answer:
        """
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain

    def save_answers_to_excel(self, questions, answers, output_path):
        df = pd.DataFrame({'Questions': questions, 'Answers': answers})
        df.to_excel(output_path, index=False)