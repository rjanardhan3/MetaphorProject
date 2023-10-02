import os
import openai
from metaphor_python import Metaphor
from openai.embeddings_utils import get_embedding, cosine_similarity
import pandas as pd
import numpy as np
import spacy
from ast import literal_eval
import textstat

# Initializing Spacy Library and requisite API keys from environment
openai.api_key = os.getenv("OPENAI_API_KEY")
nlp = spacy.load('en_core_web_lg')
metaphor = Metaphor(os.getenv("METAPHOR_API_KEY"))

#Grabbing prompt and getting the LLM Response
prompt = input("Give me a prompt to evaluate:")

USER_QUESTION = prompt
N = 3

SYSTEM_MESSAGE_QUERY = "You are a helpful assistant that generates a search query based on a given prompt."
get_completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": prompt},
    ]
)

#Getting the actual LLM Response and Outputting
llm_response = get_completion["choices"][0]["message"]["content"]
print(get_completion["choices"][0]["message"]["content"])

#use Metaphor to get the corresponding websites
query = get_completion["choices"][0]["message"]["content"]
search_response = metaphor.search(
    query, use_autoprompt=True, start_published_date="2023-06-01"
)

SYSTEM_MESSAGE = "You are a helpful assistant that summarizes the content of a webpage. Summarize the users input."
SYSTEM_MESSAGE_EVAL = "You are a helpful assistant that takes in a prompt and response pair and gives the response a rating from 1 to 8, without any extra words or commas"
contents_result = search_response.get_contents()

#Setting up variables
summaries = []
flesch_reading_ease_score = 0.0
gunning_fog_score = 0.0
ratings = []
rating_list = [1500] * N

#Given an LLM Response for evaluation, we can get the rating
def find_first_integer(string):
    for char in string:
        if char.isdigit():
            return int(char)
    return 1

#Iterate through N results from Metaphor
for i in range(N):
    #Get the summary because of request length issues
    first_result = contents_result.contents[i]
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": first_result.extract},
        ],
    )

    #Get specific evaluation metrics and store them
    summary = completion.choices[0].message.content
    summaries.append(completion.choices[0].message.content)
    current_evaluation = []
    flesch_reading_ease_score += textstat.flesch_reading_ease(summary)
    gunning_fog_score += textstat.gunning_fog(summary)

    #Get evaluation ratings and store them
    eval_prompt = "Prompt:\n"
    eval_prompt += prompt + "\n"
    eval_prompt += "Response:\n"
    eval_prompt += summary
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_MESSAGE_EVAL},
            {"role": "user", "content": eval_prompt},
        ]
    )
    ratings.append(find_first_integer(completion.choices[0].message.content))

#Get the Net summary of all of the summaries
net_summary = ""
for i in range(N):
    net_summary += summaries[i]
    net_summary += "/n"

#calculate the net summary and llm response stats
website_answer = net_summary
llm_gunning_fog = textstat.gunning_fog(llm_response)
llm_flesch_reading_score = textstat.flesch_reading_ease(llm_response)

### SIMILARITY AND EVALUATION METRICS:
doc1 = nlp(website_answer)
doc2 = nlp(llm_response)
similarity_score = doc1.similarity(doc2)
print("\nSimilarity Score of LLM Response with respect to queried websites: ", similarity_score)
flesch_reading_ease_score /= N
gunning_fog_score /= N
print("\nAverage Gunning Fog Score of Queried Websites: ", gunning_fog_score)
print("\nLLM Gunning Fog Score: ", llm_gunning_fog)
print("\nAverage Flesch Reading Score of Queried Websites: ", flesch_reading_ease_score)
print("\nLLM Average Flesch Reading score: ", llm_flesch_reading_score)

#Calculates the ELO updates according to the 40-update rule
def calculate_elo_update(team_a_score, team_b_score, winner):
    rating_diff = (team_b_score - team_a_score)
    rating_ratio = rating_diff/400
    expected_update = 1/(1 + pow(10, (rating_ratio)))
    team_a_score += 40*(winner - expected_update)
    if winner == 1:
        winner = 0
    elif winner == 0:
        winner = 1
    team_b_score += 40*(winner - expected_update)

    return (team_a_score, team_b_score)

#Goes through iteratively and creates an ELO dashboard for the URLs
for i in range(N):
    for j in range(i+1, N):
        val = 0.5
        if ratings[i] > ratings[j]:
            val = 0
        elif ratings[i] < ratings[j]:
            val = 1
        (val1, val2) = calculate_elo_update(rating_list[i], rating_list[j], val)
        rating_list[i] = val1
        rating_list[j] = val2

    print("\nELO Rating for Website: ", search_response.results[i].url, "\n", rating_list[i])
