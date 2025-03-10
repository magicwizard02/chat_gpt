{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from langchain.chat_models import ChatOpenAI\n",
    "from langchain.prompts import ChatPromptTemplate\n",
    "from langchain.callbacks import StreamingStdOutCallbackHandler\n",
    "\n",
    "# Define Chat Model\n",
    "chat = ChatOpenAI(model_name=\"gpt-3.5-turbo\", temperature=0.1, streaming = True, callbacks=[StreamingStdOutCallbackHandler(),])\n",
    "\n",
    "# Poet Chain (Generates a Haiku)\n",
    "poet_prompt = ChatPromptTemplate.from_messages([\n",
    "    (\"system\", \"You are specialized in writing {literature} about programming languages that is easy and sophisticated to understand.\"),\n",
    "    (\"human\", \"I want to write a {literature}.\")\n",
    "])\n",
    "poet_chain = poet_prompt | chat   # Parses output into string format\n",
    "\n",
    "# Critique Chain (Explains the Haiku)\n",
    "critique_prompt = ChatPromptTemplate.from_messages([\n",
    "    (\"system\", \"You are specialized in explaining {poem}.\"),\n",
    "    (\"human\", \"I want to explain a {poem}.\")\n",
    "])\n",
    "critique_chain = critique_prompt | chat \n",
    "\n",
    "# Final Chain (Generate Haiku → Explain it)\n",
    "final_chain = {\"poem\": poet_chain} | critique_chain\n",
    "\n",
    "# Invoke the chain\n",
    "response = final_chain.invoke({\n",
    "    \"literature\": \"Haikus\"\n",
    "})\n",
    "\n",
    "# Print result\n",
    "print(response)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
