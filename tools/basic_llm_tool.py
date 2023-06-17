
from langchain.tools import BaseTool
from math import pi
from typing import Union


class BasicLLMTool(BaseTool):
    name = "General conversation"
    description = "use this tool for general conversational interactions on topics that do not need in-depth knowledge."

    def _run(self, llm_chain):
        return 
    
    def _arun(self, radius: Union[int, float]):
        raise NotImplementedError("This tool does not support async")