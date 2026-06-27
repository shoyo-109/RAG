from openai.types import model
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core import __version__ as core_version
from langchain import __version__  as lg_version
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI



def main():

    # testing Gemini
    #   

    # testing NVIDIA Nemotron Omni Reasoning Model
    llm_nvidia = ChatOpenAI(
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
    response_nv = llm_nvidia.invoke("Hello, consider yourself as a 20 year old male from india, How do you define a female friend ?")
    # To view the reasoning tokens in LangChain if returned
    reasoning = response_nv.additional_kwargs.get("reasoning_content")
    if reasoning:
        print("Nemotron Reasoning:\n", reasoning)
    print("Nemotron response: ", response_nv.content)
    print("\n")
    

    # # testing CHatGPT
    # llm_chatgpt = ChatOpenAI(model_name="gpt-4o-mini",temperature=0,api_key=os.getenv("OPENAI_API_KEY"))
    # response=llm_chatgpt.invoke("Hello, How do you define a Women ?")
    # print("ChatGPT response: ",response.content)

if __name__ == "__main__":
    main()
