import re
import ast
import sys
import base64
import os
from typing import List, Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from .crawler import scrape_text_crawl4ai as crawl

def clean_text(text: str) -> str:
    """Removes <think> tags from model output."""
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned.strip()

def extract_code(text: str, lang: str = "python") -> str:
    """Extracts code from markdown blocks."""
    pattern = r"```" + lang + r"(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    if "```" in text:
        return text.split("```")[1].strip()

    return text.strip()

def run_llm(model_name: str, user_prompt: str, system_prompt: str = None, temperature: float = 0.1) -> str:
    llm = ChatOllama(model=model_name, temperature=temperature)

    messages = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=user_prompt))

    try:
        response = llm.invoke(messages)
        return clean_text(response.content)
    except Exception as e:
        return f"LLM Error: {str(e)}"

def run_vision_llm(model_name: str, prompt_text: str, image_path: Optional[str], temperature: float = 0.1) -> str:
    llm = ChatOllama(model=model_name, temperature=temperature)

    content_parts = [{"type": "text", "text": prompt_text}]

    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')

        content_parts.append({
            "type": "image_url",
            "image_url": f"data:image/png;base64,{image_data}"
        })

    msg = HumanMessage(content=content_parts)
    try:
        response = llm.invoke([msg])
        return clean_text(response.content)
    except Exception as e:
        return f"Vision LLM Error: {str(e)}"

def get_third_party_imports(code: str) -> List[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_modules.add(node.module.split('.')[0])

    std_lib = sys.stdlib_module_names if hasattr(sys, 'stdlib_module_names') else set(sys.builtin_module_names)
    third_party = [m for m in imported_modules if m not in std_lib]
    return list(third_party)
