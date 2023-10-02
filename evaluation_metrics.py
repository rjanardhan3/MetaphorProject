import os
import openai
from metaphor_python import Metaphor
from openai.embeddings_utils import get_embedding, cosine_similarity
import pandas as pd
import numpy as np
import spacy
from ast import literal_eval
import textstat

#Load Spacy and get requisite API keys
openai.api_key = os.getenv("OPENAI_API_KEY")
metaphor = Metaphor(os.getenv("METAPHOR_API_KEY"))
nlp = spacy.load('en_core_web_lg')

#get prompt and the requisite LLM Response
prompt = input("Give me a prompt to evaluate:")

SYSTEM_MESSAGE_EVAL = "You are a helpful assistant that takes in a prompt and response pair and gives the response a rating from 1 to 8, without any extra words or commas"

USER_QUESTION = prompt
N = 3

#get the response from the LLM
SYSTEM_MESSAGE_QUERY = "You are a helpful assistant that generates a search query based on a given prompt."
get_completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": prompt},
    ]
)

#function to get first integer from an LLM Response
def find_first_integer(string):
    for char in string:
        if char.isdigit():
            return int(char)
    return 1

#function to calculate the similarity(Spacy calculates in the embedding space)
def find_similarity(a, b):
    doc1 = nlp(a)
    doc2 = nlp(b)
    return doc1.similarity(doc2)

llm_response = get_completion["choices"][0]["message"]["content"]

query = get_completion["choices"][0]["message"]["content"]
search_response = metaphor.search(
    query, use_autoprompt=True,
)

SYSTEM_MESSAGE = "You are a helpful assistant that summarizes the content of a webpage. Summarize the users input."
contents_result = search_response.get_contents()

#gets the summary given a message
def get_summary(first_result):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": first_result},
        ],
    )
    summary = completion.choices[0].message.content
    return summary

#iterates through N of the responses from Metaphor
for i in range(N):
    first_result = contents_result.contents[i]
    base_summary = get_summary(first_result.extract)
    current_evaluation = find_similarity(llm_response, base_summary)

    urls = []
    similarResponse = metaphor.find_similar(search_response.results[i].url)
    for j in range(len(similarResponse.results)):
        additional_summary = get_summary(metaphor.get_contents(similarResponse.results[j].id).contents[0].extract)
        eval = find_similarity(llm_response, additional_summary)
        if (eval > current_evaluation):
            urls.append(similarResponse.results[j].url)

    #Print out if the URLs are better or not
    if len(urls) == 0:
        print("\nNone of the similar links for ", search_response.results[i].url, " are better")
    else:
        print("\nHere are a list of more comprehensive links for ", search_response.results[i].url)
        for j in range(len(urls)):
            print("\n", urls[j])
