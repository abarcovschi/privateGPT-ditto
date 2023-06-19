from dotenv import load_dotenv
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, LLMChain
from langchain.tools import DuckDuckGoSearchRun 
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
import re
import langchain
load_dotenv()

# Define which tools the agent can use to answer user queries
search = DuckDuckGoSearchRun()


def duck_wrapper(input_text):
    search_results = search.run(f"site:webmd.com {input_text}")
    return search_results

def google_wrapper(input_text):
    search_results = search.run(f"site:google.com {input_text}")
    return search_results

tools = [
    Tool(
        name = "Search WebMD",
        func=duck_wrapper,
        description="useful for when you need to answer medical and pharmalogical questions"
    )
    # ,
    # Tool(
    #     name = "Google Search",
    #     func= google_wrapper,
    #     description="useful for when you need to search google for information"
    # )
]

# Set up the base template
# template = """Answer the following questions as best you can, but speaking as compasionate medical professional. You have access to the following tools:

# {tools}

# Use the following format:

# Question: the input question you must answer
# Thought: you should always think about what to do
# Action: the action to take, should be one of [{tool_names}]
# Action Input: the input to the action
# Observation: the result of the action
# ... (this Thought/Action/Action Input/Observation can repeat N times)
# Thought: I now know the final answer
# Final Answer: the final answer to the original input question

# Begin! Remember to answer as a compansionate medical professional when giving your final answer.

# Question: {input}
# {agent_scratchpad}"""

# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]
    
    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)
    
# prompt = CustomPromptTemplate(
#     template=template,
#     tools=tools,
#     # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
#     # This includes the `intermediate_steps` variable because that is needed
#     input_variables=["input", "intermediate_steps"]
# )

class CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
output_parser = CustomOutputParser()

llm = OpenAI(temperature=0)

# LLM chain consisting of the LLM and a prompt
# llm_chain = LLMChain(llm=llm, prompt=prompt)
# tool_names = [tool.name for tool in tools]

# agent = LLMSingleActionAgent(
#     llm_chain=llm_chain, 
#     output_parser=output_parser,
#     stop=["\nObservation:"], 
#     allowed_tools=tool_names
# )
# agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, 
#                                                     tools=tools, 
#                                                     verbose=True)
#agent_executor.run("How can I treat a sore throat?")

#Conversation Memory
# Set up the base template
template_with_history = """Answer the following questions as best you can, but speaking as compasionate medical professional. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Remember to speak as a compasionate medical professional when giving your final answer. If the condition is serious advise they speak to a doctor.

Previous conversation history:
{history}

New question: {input}
{agent_scratchpad}"""

prompt_with_history = CustomPromptTemplate(
    template=template_with_history,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps", "history"]
)
llm_chain = LLMChain(llm=llm, prompt=prompt_with_history) 
tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain, 
    output_parser=output_parser,
    stop=["\nObservation:"], 
    allowed_tools=tool_names
)

from langchain.memory import ConversationBufferWindowMemory

#K= number of messages to keep track of
memory=ConversationBufferWindowMemory(k=2)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, 
    tools=tools, 
    verbose=True, 
    memory=memory
    )

agent_executor.run("How do I treat a sore throat?")
agent_executor.run("What medicine should I take?")
#agent_executor.run("How long will it take to heal?")