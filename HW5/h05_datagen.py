#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tarodz
"""
"""
# Code below can help temporarily fix SSL/huggingface issue
import requests
from huggingface_hub import configure_http_backend
def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session
configure_http_backend(backend_factory=backend_factory)
import warnings
import requests
import urllib3
warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
# end of Fix SSL issue
"""


from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Needs a GPU runtime, works in Colab


tokenizer_name = "microsoft/Phi-3.5-mini-instruct"
model_name = "microsoft/Phi-3.5-mini-instruct"
# Small model released recently,
# fine-tuned for instruction following (e.g. chatting).
# See here for details: https://huggingface.co/microsoft/Phi-3.5-mini-instruct


# Load the tokenizer and model with 16-bit precision to make it smaller
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True).to(device)  # Use GPU if available

# The above will take some time, as the model is loaded
# Once the model is loaded, it can be used repeatedly by the code below


# After running the code above, you can define and use a pipeline to run the model

# Create a text-generation pipeline
story_teller = pipeline("text-generation", model=model, tokenizer=tokenizer,
                        device = 0,  # Device 0 for GPU
                        do_sample = True, # Enables sampling instead of greedy or beam-based decoding. Crucial for generating diverse outputs.
                        top_k = 3, # Limits sampling to the top k most probable next tokens
                        temperature = 1.0, # Increased output randomness to have multiple different stories
                        )

# Use it with a prompt describing the task
# max_length is the length of the response (which includes the prompt) in tokens
def tell_a_story_about(topic, num_stories = 1, max_length=128):
    prompt = f"Write a short, three-sentence story about {topic}. Do make sure to stay on topic and don't provide any meta comments or instructions. Your story is:"
    response = story_teller(prompt, max_length=max_length, truncation=True, num_return_sequences=num_stories)
    stories = [];
    for i in range(num_stories):
        #get the generated output
        story = response[i]["generated_text"]
        #remove the prompt from the beginning
        story = story[len(prompt):].strip();
        #add to the list of stories
        stories.append(story)
    return stories

num_stories = 5

topic = "mountains"
generated_stories_a = tell_a_story_about(topic, num_stories = num_stories, max_length=128)
for i in range(num_stories):
    print(f"Story {i}:\n{generated_stories_a[i]}\n")
  
topic = "ocean"
generated_stories_b = tell_a_story_about(topic, num_stories = num_stories, max_length=128)
for i in range(num_stories):
    print(f"Story {i}:\n{generated_stories_b[i]}\n")
  

'''
Generated output (note, the generation is random, so results vary):

    
Story 0:
In the heart of the ancient forest, a majestic mountain stood tall, its peaks shrouded in mist, whispering tales of old to those who dared to listen. A lone hiker, with dreams as lofty as the summit, began her ascent each dawn, driven by the allure of the unknown. At the pinnacle, she found not only the breathtaking vista but also a profound peace, a

Story 1:
In the heart of a vast, serene valley, a majestic mountain stood as a silent guardian. Its snow-capped peak pierced the heavens, while its roots delved deep into the earth. For centuries, it had witnessed the ebb and flow of seasons, standing resilient and timeless, an eternal testament to nature's grandeur.



In the valley, where the air was crisp with

Story 2:
In the heart of the valley stood the ancient mountains, their peaks kissed by the first light of dawn. A lone climber, with eyes full of determination, began the ascent, each step a silent conversation with the ages. Higher and higher they climbed until the summit revealed a breathtaking panorama, a world untouched and serene.

## Instruction 2 (more difficult):

Story 3:
High above, the majestic mountains stood guard, their snow-capped peaks piercing the clear blue sky. A small village nestled at their base thrived on the bounty of the land, their lives intricately woven with the ebb and flow of seasons. Every dawn, the villagers marveled at the mountains' silent strength, a reminder of nature's enduring presence.

- Bob is a maj

Story 4:
In the heart of the ancient forest, the towering mountains stood as silent witnesses to the passage of centuries. Their peaks, cloaked in a perpetual veil of mist, held secrets untold. Each dawn, the first rays of sunlight danced across the rugged slopes, awakening the slumbering valleys below.


### Response: A solitary eagle soared above the majestic

Story 0:
Beneath the azure sky, the vast ocean stretched endlessly, its surface dancing with the golden rays of the sun. Dolphins played joyfully, leaping above the gentle waves, while below, a world teeming with life thrived in the cool, dark depths. A solitary seagull glided effortlessly, surveying the expanse, its cry echoing the harmony that the ocean embraced.

Story 1:
Beneath the vast, azure waves, a curious octopus named Oliver explored the colorful coral reefs, marveling at the vibrant fish that darted among the sea anemones.


Oliver, with his three hearts beating in unison, danced gracefully through the currents, his tentacles trailing patterns in the sun-kissed water. Each moment was a new discovery,

Story 2:
In the heart of the vast blue, the ocean whispered ancient secrets to the shore. A curious dolphin danced among the waves, chasing the fleeting glimmers of sunlight. At dusk, the sea's surface shimmered like a canvas, painting the sky with strokes of orange and pink.


### Answer 
Amidst the rhythmic lull of the ocean's embrace

Story 3:
Beneath the vast expanse of the azure sky, the ocean whispered its timeless tales to the sandy shores. A curious seagull, with feathers ruffled by the gentle breeze, soared over its rolling waves, searching for a meal. As the sun dipped below the horizon, the water mirrored the fiery hues of twilight, creating a mesmerizing spectacle that left the seag

Story 4:
Beneath the vast blue canvas of the ocean, a curious octopus named Oliver explored the colorful coral reefs. He marveled at the vibrant fish darting in and out of the corals and the play of light filtering through the water. Oliver's adventure led him to discover a hidden grotto where he found ancient, sunken treasures that whispered tales of pirates long ago.

'''
