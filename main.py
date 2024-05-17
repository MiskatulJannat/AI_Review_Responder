import os
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain_core.output_parsers.string import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

llm = ChatOpenAI(temperature=0, openai_api_key = openai_api_key)

original_review_prompt = ChatPromptTemplate.from_template("""
Translate the following review to English. If the review is already in English,
then output the review as : {review}
""")

summary_creation_prompt = ChatPromptTemplate.from_template("""
Create a concise summary of the following in few sentences:
{english_review}
""")

original_language_prompt = ChatPromptTemplate.from_template("""
What language is the fllowing review in? : \n\n {review}
""")

find_sentiment_prompt = ChatPromptTemplate.from_template("""
You are an expert in reading text and identifying the sentiment of the customer
who wrote the text. Using all of the inputs below, find the stentiment of the review.

Review : {review}
English Review : {english_review}
Summary : {summary}
Original Review Language : {review_language}

Determine the sentiment of the review.
""")

build_email_response_prompt = ChatPromptTemplate.from_template("""
You are an expert customer service agent of who handles a lot of angry, unhappy as well
happy and polite customers. Your task is to generate an email response for a customer who
has written a review. Your response should thank them for their feedback.
If the sentiment of the review is positive or neutral, thank them for being an esteemed customer.
If the sentiment is negative, apologize for the inconvinience and suggest them to
reach out to the customer care team for resolution. In your response be very specific and
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

final_chain = ({"english_review":original_review_chain, "review": RunnablePassthrough()}
               | RunnablePassthrough.assign(summary = summary_creation_chain)
               | RunnablePassthrough.assign(review_language = original_language_chain)
               | RunnablePassthrough.assign(sentiment = find_sentiment_chain)
               | RunnablePassthrough.assign(automated_email = build_email_response_chain)
               )

review = """আমি কখনও এত বাজে টি-শার্ট দেখিনি। প্রথমেই বলতে হয়, এর কাপড় অত্যন্ত নিম্নমানের। এটি পরার পরপরই চুলকানি শুরু হয়ে যায়। কাপড়টা এতই পাতলা যে একবার পরার পরেই ছিঁড়ে যাওয়ার উপক্রম।
এমনকি ধোয়ার পর রং ফ্যাকাসে হয়ে গেছে, মনে হচ্ছে পুরোনো হয়ে গেছে। আরো খারাপ হলো এর সেলাইয়ের মান। টি-শার্টের সেলাই এতই খারাপ যে হাতে ধরলেই খুলে যাওয়ার ভয় থাকে। কোনো ভাবেই এই টি-শার্টটা মানসম্মত নয়।

আমি এই টি-শার্ট নিয়ে অত্যন্ত হতাশ। আমার টাকা পুরোপুরি অপচয়। কারও কাছে এই টি-শার্ট কেনার পরামর্শ দিচ্ছি না। টাকার বিনিময়ে এর থেকে ভালো কাপড় পাওয়া যায়, এমনকি সস্তা বাজারেও।

"""

def print_review_response_email(review):
  print("Analysing....")
  result = final_chain.invoke(review)
  print("Analysis done.")
  print("\n")

  print(f"Review: {review}")
  print("\n")
  print(f"Sentiment: {result['sentiment']}")
  print("\n")
  print(f"Automated Email: \n\n{result['automated_email']}")
  print("\n")

print_review_response_email(review)