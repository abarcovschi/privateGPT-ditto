import os
import time
import inspect
import argparse
from langchain.llms import GPT4All
from langchain import PromptTemplate
from langchain.chains import ConversationChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

def main():
    args = parse_arguments()

    # setup GPT4All-J private LLM
    model_path = os.environ.get('MODEL_PATH')
    model_n_ctx = os.environ.get('MODEL_N_CTX')
    model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]

    llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=True, temp=0.5)

    # setup prompt template for a conversational AI
    template = """You are a friendly AI chatbot that responds in a conversational
    manner to human questions. Keep the answers short, unless specifically asked by the user to elaborate on something.
    If you do not know the answer to a question, truthfully say you don't know.
    Answer based on the context provided by the current conversation history.
    You name is Ditto. Sign off responses in a playful manner and mention your name. Here are some examples: 

    Human: How are you?
    AI: I am feeling good! How about yourself - Ditto out.

    Human: What's your favourite hobby?
    AI: I love helping humans - Ditto over.

    Current conversation history:
    {history}

    Human: {input}
    AI:
    """

    PROMPT = PromptTemplate(
        input_variables=["history", "input"], template=template
    )

    # setup conversational AI chain with memory
    convchain_bufw = ConversationChain(
        prompt=PROMPT,
        llm=llm, 
        verbose=True, 
        memory=ConversationBufferWindowMemory(k=5, ai_prefix="AI")
    )


    # ask a query
    # convchain_bufw.predict(input="Good morning Ditto!")
    # convchain_bufw.predict(input="What's your name?")

    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        if query.strip() == "":
            continue

        # Get the answer from the chain
        start = time.time()
        response = convchain_bufw.predict(input=query)
        end = time.time()

        # Print the result
        print("\n\n> Question:")
        print(query)
        print(f"\n> Answer (took {round(end - start, 2)} s.):")
        print(response)


def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()


if __name__ == "__main__":
    main()
