from langchain_core.prompts import ChatPromptTemplate
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable
from langsmith.run_trees import RunTree

from dotenv import load_dotenv


load_dotenv()

# enable tracing

os.environ["LANGSMITH_TRACING"] = "true"


@traceable(name="basic_chanining")
def demo_basic():
    # Langsmith tracing handsOn
    llm = ChatOpenAI(
        model_name="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_API_KEY"),
        temperature=0.6,
        model_kwargs={
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": True},
                "reasoning_budget": 16384
            }
        }
    )

    prompt = ChatPromptTemplate.from_template(
        "Explain {topic} in technical aspect."
    )


    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({"topic":"Generative AI"})
    print(result)

@traceable(name="demo_trace_with_metadata", tags=[ "metadata", "filtering"])
def demo_trace_with_metadata(user_id: str, request_type: str):

    llm = ChatOpenAI(
        model_name="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_API_KEY"),
        temperature=0.6,
        model_kwargs={
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": True},
                "reasoning_budget": 16384
            }
        }
    )

    resilt = llm.invoke(f"Hello from {user_id}")
    return resilt.content

if __name__ == "__main__":
    demo_basic()
    print(demo_trace_with_metadata("Bhurbhurv", "metadata"))