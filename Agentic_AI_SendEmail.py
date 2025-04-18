from langchain.agents import initialize_agent, Tool,AgentExecutor
import os
from langchain.llms.base import LLM 
import openvino_genai as ov_genai   
from typing import List,Optional

model_path = 'Phi-35_mini_instruct_refined'
pipe = ov_genai.LLMPipeline(model_path, device='NPU')

class OpenVINO_LLM(LLM):
    def __init__(self,pipeline: ov_genai.LLMPipeline,**kwargs):
        super().__init__(**kwargs)
        self._pipeline = pipeline
    @property
    def pipeline(self) -> ov_genai.LLMPipeline:
        return self._pipeline
    @property
    def _llm_type(self) -> str:
        return "openvino_genai"
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = []
        def capture_output(subword):
            response.append(subword)
            return False
        self.pipeline.generate(prompt, streamer=capture_output)
        return "".join(response)
openvino_llm = OpenVINO_LLM(pipeline=pipe)
