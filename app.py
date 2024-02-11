import os
from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

load_dotenv(find_dotenv())

# img2text

def img2text(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

    text = image_to_text(url)[0]["generated_text"]

    print(text)
    return text

def generate_story(scenario):
    template = """
    You are a story teller;
    You can generate a short story based on a simple narrative, the story should be no more than 20 words;

    CONTEXT: {scenario}
    STORY:
    """
    prompt = PromptTemplate(template=template, input_variables=["scenario"])

    story_llm = LLMChain(llm=ChatOpenAI(
        model_name="gpt-3.5-turbo"), prompt=prompt)

    story = story_llm.predict(scenario=scenario)

    print(story)
    return story

scenario = img2text("dale.png")
story = generate_story(scenario)