import asyncio
import json
import ollama
from tqdm import tqdm
import pandas as pd
# Import đúng schema theo document bạn cung cấp
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness, LLMContextPrecisionWithoutReference, LLMContextRecall, ResponseRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.llms import Ollama

from typing import List
from langchain_core.embeddings import Embeddings

class OllamaQwen3Embeddings(Embeddings):
    def __init__(self, model_name: str = "qwen3-embedding:0.6b"):
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in tqdm(texts):
            res = ollama.embed(model=self.model_name, input=text)
            embeddings.append(res['embeddings'][0])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        res = ollama.embed(model=self.model_name, input=text)
        return res['embeddings'][0]

async def main():
    model_answers = [
      {
        "test_case_id": "TC_PSE1.1_01",
        "requirement_id": "PSE1.1",
        "scenario_id": "PSE1.1-S1",
        "title": "Successful Registration",
        "preconditions": ["User is not registered in the system"],
        "steps": [
          "Navigate to the registration page",
          "Enter a valid email address in the Email field",
          "Enter a strong password in the Password field",
          "Click the Register button"
        ],
        "test_data": {
          "email": "newuser@example.com",
          "password": "StrongPass!123"
        },
        "expected_result": "User account is created successfully and the user is redirected to the dashboard",
        "priority": "High",
        "is_automated": "true"
    },
    {
        "test_case_id": "TC_PSE1.1_02",
        "requirement_id": "PSE1.1",
        "scenario_id": "PSE1.1-S2",
        "title": "Email Format Validation - Invalid Email",
        "preconditions": ["User is not registered in the system"],
        "steps": [
          "Navigate to the registration page",
          "Enter an invalid email address (e.g., 'invalidemail') in the Email field",
          "Enter a strong password in the Password field",
          "Click the Register button"
        ],
        "test_data": {
          "email": "invalidemail",
          "password": "StrongPass!123"
        },
        "expected_result": "Error message displayed: 'Please enter a valid email address'",
        "priority": "High",
        "is_automated": "true"
    },
    {
        "test_case_id": "TC_PSE1.1_03",
        "requirement_id": "PSE1.1",
        "scenario_id": "PSE1.1-S3",
        "title": "Password Validation - Weak Password",
        "preconditions": ["User is not registered in the system"],
        "steps": [
          "Navigate to the registration page",
          "Enter a valid email address in the Email field",
          "Enter a weak password (e.g., '123') in the Password field",
          "Click the Register button"
        ],
        "test_data": {
          "email": "newuser@example.com",
          "password": "123"
        },
        "expected_result": "Error message displayed: 'Password does not meet security requirements'",
        "priority": "High",
        "is_automated": "true"
    },
    {
        "test_case_id": "TC_PSE1.1_04",
        "requirement_id": "PSE1.1",
        "scenario_id": "PSE1.1-S4",
        "title": "Duplicate Email Check",
        "preconditions": ["A user account with email 'existinguser@example.com' already exists"],
        "steps": [
          "Navigate to the registration page",
          "Enter the duplicate email address 'existinguser@example.com' in the Email field",
          "Enter a strong password in the Password field",
          "Click the Register button"
        ],
        "test_data": {
          "email": "existinguser@example.com",
          "password": "StrongPass!123"
        },
        "expected_result": "Error message displayed: 'Email already registered'",
        "priority": "High",
        "is_automated": "true"
    }
    ]
    
    native_ollama = Ollama(
        model="llama3.1:8b", 
        base_url="http://localhost:11434"
    )
    evaluator_llm = LangchainLLMWrapper(native_ollama)
    
    # Tạo custom embeddings instance
    native_embeddings = OllamaQwen3Embeddings(model_name="qwen3-embedding:0.6b")

    # Wrap nó để Ragas dùng được (rất quan trọng!)
    evaluator_embeddings = LangchainEmbeddingsWrapper(native_embeddings)

    faithfull_scorer = Faithfulness(llm=evaluator_llm)
    context_precision_scorer = LLMContextPrecisionWithoutReference(llm=evaluator_llm)
    context_recall_scorer = LLMContextRecall(llm=evaluator_llm)
    respone_relevancy_scorer = ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)

    try:
        df = pd.read_csv(r"D:\RAG_testcases\sample_eval_data.csv")
    except Exception as e:
        print(f"Lỗi đọc file: {e}")
        return
    model_answer_str = json.dumps(model_answers, ensure_ascii=False)

    sample = SingleTurnSample(
        user_input=str(df.loc[0, "query"]),
        response=model_answer_str,
        retrieved_contexts=[str(df.loc[0, "retrieved_context"])]
    )
    
    sameple_context_recall = SingleTurnSample(
        user_input=str(df.loc[0, "query"]),
        response=model_answer_str,
        reference=str(df.loc[0, "reference_answer"]),
        retrieved_contexts=[str(df.loc[0, "retrieved_context"])]
    )

    print("--- Đang đánh giá dựa trên SingleTurnSample ---")
    try:
        faithfull = await faithfull_scorer.single_turn_ascore(sample)
        context_precision_score = await context_precision_scorer.single_turn_ascore(sample)
        context_recall_score = await context_recall_scorer.single_turn_ascore(sameple_context_recall)
        respone_relevancy_score = await respone_relevancy_scorer.single_turn_ascore(sample)
        
        print(f"\nFaithfulness Score: {faithfull: .2f}")
        print(f"\nContext Precision Score: {context_precision_score: .2f}")
        print(f"\nContext Recall Score: {context_recall_score: .2f}")
        print(f"\nResponse Relevancy Score: {respone_relevancy_score: .2f}")
        
    except Exception as e:
        print(f"Gặp lỗi khi đánh giá: {e}")

if __name__ == "__main__":
    asyncio.run(main())