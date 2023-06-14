from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import GPT4All
import os
import argparse
import inspect

from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import (ConversationBufferMemory, 
                                                  ConversationSummaryMemory, 
                                                  ConversationBufferWindowMemory,
                                                  ConversationKGMemory)#

def main():
    args = parse_arguments()

    model_path = os.environ.get('MODEL_PATH')
    model_n_ctx = os.environ.get('MODEL_N_CTX')
    model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]

    llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)


    conversation = ConversationChain(
        llm=llm, 
    )

    conversation_bufw = ConversationChain(
        llm=llm, 
        memory=ConversationBufferWindowMemory(k=5)
    )
    query = "Good morning AI!"
    conversation_bufw.run(query)
    a=1


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
