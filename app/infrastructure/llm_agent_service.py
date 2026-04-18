from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_qwq import ChatQwen
from langchain_openrouter import ChatOpenRouter

import os
from dotenv import load_dotenv
from typing import Literal
load_dotenv()


def model_init(
    model_name: str = "gemini-2.5-flash",
    type_api: Literal["gemini", "huggingface", "openai", "qwen", "openrouter"] = "gemini",
    temperature: float = 0
):
    match type_api:
        case "gemini":
            chat_model = ChatGoogleGenerativeAI(model=model_name, temperature=temperature, name=model_name)
        case "huggingface":
            llm_config = HuggingFaceEndpoint(
                repo_id=model_name,
                task="text-generation",
                provider="auto",
                temperature=temperature,
                max_new_tokens=10000
            )
            chat_model = ChatHuggingFace(llm=llm_config, name=model_name)
        case "openai":
            chat_model = ChatOpenAI(model=model_name, temperature=temperature, name=model_name)
        case "qwen":
            chat_model = ChatQwen(model=model_name, temperature=temperature, api_key=os.getenv("DASHCOPE_API_KEY"), name=model_name)
        case "openrouter":
            chat_model = ChatOpenRouter(model=model_name, temperature=temperature, app_title=None)
        case _:
            return None
    return chat_model