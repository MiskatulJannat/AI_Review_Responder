import os
from dotenv import load_dotenv
import pickle

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain_core.output_parsers.string import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)

original_review_prompt = ChatPromptTemplate.from_template("""
Translate the following review to English. If the review is already in English,
then output the review as : {review}
""")

summary_creation_prompt = ChatPromptTemplate.from_template("""
Create a concise summary of the following in few sentences:
{english_review}
""")

original_language_prompt = ChatPromptTemplate.from_template("""
What language is the following review in? : \n\n {review}
""")

find_sentiment_prompt = ChatPromptTemplate.from_template("""
You are an expert in reading text and identifying the sentiment of the customer
who wrote the text. Using all of the inputs below, find the sentiment of the review.

Review : {review}
English Review : {english_review}
Summary : {summary}
Original Review Language : {review_language}

Determine the sentiment of the review.
""")

build_email_response_prompt = ChatPromptTemplate.from_template("""
You are an expert customer service agent who handles a lot of angry, unhappy as well
as happy and polite customers. Your task is to generate an email response for a customer who
has written a review. Your response should thank them for their feedback.
If the sentiment of the review is positive or neutral, thank them for being an esteemed customer.
If the sentiment is negative, apologize for the inconvenience and suggest they
reach out to the customer care team for resolution. In your response, be very specific and
informative by using the specific details from the customer feedback.
Your response should be concise and professional.
Sign the email with your name as "Customer Support Team".

Review : {review}
English Review : {english_review}
Summary : {summary}
Original Review Language : {review_language}
Sentiment : {sentiment}
""")

original_review_chain = original_review_prompt | llm | StrOutputParser()
summary_creation_chain = summary_creation_prompt | llm | StrOutputParser()
original_language_chain = original_language_prompt | llm | StrOutputParser()
find_sentiment_chain = find_sentiment_prompt | llm | StrOutputParser()
build_email_response_chain = build_email_response_prompt | llm | StrOutputParser()

final_chain = (
    {"english_review": original_review_chain, "review": RunnablePassthrough()}
    | RunnablePassthrough.assign(summary=summary_creation_chain)
    | RunnablePassthrough.assign(review_language=original_language_chain)
    | RunnablePassthrough.assign(sentiment=find_sentiment_chain)
    | RunnablePassthrough.assign(automated_email=build_email_response_chain)
)

# Save the final_chain to a pickle file
with open('final_chain.pkl', 'wb') as f:
    pickle.dump(final_chain, f)
