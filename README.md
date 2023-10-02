# MetaphorProject

This is the README.md for the Metaphor Take Home Assessment.

The focus of my project was evaluation of LLMs and leveraging Metaphor's Internet capabilites.

I split my project up into 3 categories: Evaluation of LLM Responses to the Internet, Evaluations of Website Quality, and Evaluation of Similar Links.

## Evaluation of LLM Responses to the Internet

For evaluating LLM Responses to the Internet, I used OpenAI as my base. I was evaluating the gpt-3.5-turbo model. I had the user input a prompt and would query the gpt model for the response. I would then take the prompt 
and pass that through the Metaphor API to get specific links. I would then query the links and get a general summary of each of the links and concatenate them to get a general idea of the actual Internet Sentiment. Based on that, I 
was able to run 3 main evaluation metrics. The first metric was similarity, where I used Spacy's embedding-based similarity metric to get the similarity of the LLM response to the website content. Additionally, I added flesch reading
scores and gunning fog scores. These metrics desribe the readability of each of the LLM responses and the websites. This can be used to describe how different the writing styles of the LLMs and websites are.

## Evaluations of Website Quality

The next part was evaluating website quality. In today's day and age, there are lots of types of misinformation. This section assumes GPT-3.5 as the ground-truth. After getting that, I adopted a similar type of evaluation as 
OpenAI's evals, to get a specific evaluation score for each of the websites. Based on that, I created an ELO-based(which is a chess-based system) scoring system to get the score for each of the websites. After this program,
there is a leaderboard of website quality with their representative rating.

## Evaluation of Similar Links

The last part was the evaluation fo similar links. We follow the similar format, where we get the input and the response, and we treat the output of GPT-3.5 as the ground-truth. Based on this, we check the similar links to the ones
outputted to us in the Metaphor Query. If the similar links are more similar to the ground-truth, they are outputted as "more correct" than the links from the initial Metaphor query. This can in theory be used for reinforcement learning
purposes.
