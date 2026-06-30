
# AI Meeting Companion: Speech-to-Text and Meeting Summarization App

## Overview

This project is an AI-powered meeting companion that converts uploaded audio recordings into text and generates a concise summary of the key discussion points.

The application uses **OpenAI Whisper** for automatic speech recognition, **IBM watsonx / Llama 3** for language generation and summarization, and **Gradio** to provide a simple web interface for uploading audio files and viewing the generated output.

The goal of this project is to demonstrate how speech-to-text systems and large language models can be combined to support business productivity use cases such as meeting documentation, lecture summarization, interview review, and action-item extraction.

---

## Problem Statement

Business meetings often contain important decisions, insights, and follow-up tasks, but manually writing meeting notes is time-consuming and error-prone.

This project solves that problem by building a prototype that can:

* Accept an audio file from a user
* Transcribe the spoken content into text
* Send the transcript to an LLM
* Generate a structured summary of the key points
* Display the result through an interactive web interface

---

## Features

* Upload audio files through a Gradio web interface
* Convert speech to text using OpenAI Whisper
* Process longer recordings by chunking audio into smaller segments
* Summarize the transcript using an LLM through IBM watsonx
* Extract key points from business conversations
* Display the final output in a simple browser-based application


<img width="917" height="611" alt="Screenshot From 2026-06-29 18-31-12" src="https://github.com/user-attachments/assets/62997036-cbdf-419d-93b2-7d6d3c01ed50" />

---

## Tech Stack

**Programming Language**

* Python

**Machine Learning / AI**

* OpenAI Whisper
* Hugging Face Transformers
* IBM watsonx
* Meta Llama 3
* LangChain

**Application Interface**

* Gradio

**Audio Processing**

* FFmpeg

**Core Libraries**

* `transformers`
* `torch`
* `gradio`
* `langchain`
* `ibm-watson-machine-learning`

---

## System Architecture

```text
Audio File Upload
        |
        v
Gradio Web Interface
        |
        v
Whisper Speech-to-Text Pipeline
        |
        v
Generated Transcript
        |
        v
Prompt Template
        |
        v
IBM watsonx / Llama 3 LLM
        |
        v
Meeting Summary + Key Points
```
---
<img width="917" height="611" alt="Pasted image" src="https://github.com/user-attachments/assets/2ab735d5-f5d1-48b0-abc6-dfb23aefe780" />

---

## How It Works

1. The user uploads an audio file through the Gradio interface.
2. The audio file is passed into a Hugging Face automatic speech recognition pipeline.
3. OpenAI Whisper transcribes the spoken content into text.
4. The transcript is inserted into a prompt template.
5. The prompt is sent to an IBM watsonx LLM.
6. The LLM returns a structured summary highlighting the most important points from the meeting.

---

## Example Use Case

A user uploads a business meeting recording. The application transcribes the conversation and returns a summary such as:

```text
Key Points:
1. The team discussed progress on the current project milestone.
2. A delay was identified in the data collection process.
3. The next action item is to finalize the analysis report before Friday.
4. The team agreed to review the model performance metrics in the next meeting.
```

---

## Project Structure

```text
.
├── simple_speech2text.py      # Basic Whisper speech-to-text script
├── speech2text_app.py         # Gradio app for audio transcription
├── simple_llm.py              # Simple LLM text generation script
├── speech_analyzer.py         # Full speech-to-text and summarization app
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

---

## Installation

Create and activate a virtual environment:

```bash
pip install virtualenv
virtualenv my_env
source my_env/bin/activate
```

Install the required Python libraries:

```bash
pip install --force-reinstall "setuptools<70" transformers==4.36.0 torch==2.1.1 gradio==5.23.2 langchain==0.0.343 ibm_watson_machine_learning==1.0.335 huggingface-hub==0.28.1
```

Install FFmpeg:

```bash
sudo apt update
sudo apt install ffmpeg -y
```

---

## Running the Application

Run the full meeting companion app:

```bash
python3 speech_analyzer.py
```

Then open the Gradio app in your browser at the local URL shown in the terminal.

---

## Core Implementation

```python
from transformers import pipeline
import gradio as gr
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

my_credentials = {
    "url": "https://us-south.ml.cloud.ibm.com"
}

params = {
    GenParams.MAX_NEW_TOKENS: 800,
    GenParams.TEMPERATURE: 0.1,
}

LLAMA_model = Model(
    model_id="meta-llama/llama-3-2-11b-vision-instruct",
    credentials=my_credentials,
    params=params,
    project_id="skills-network",
)

llm = WatsonxLLM(LLAMA_model)

template = """
List the key points with details from the context.

Context:
{context}
"""

prompt = PromptTemplate(
    input_variables=["context"],
    template=template
)

summary_chain = LLMChain(llm=llm, prompt=prompt)

def transcript_audio(audio_file):
    speech_to_text = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        chunk_length_s=30,
    )

    transcript = speech_to_text(audio_file, batch_size=8)["text"]
    summary = summary_chain.run(transcript)

    return summary

audio_input = gr.Audio(sources="upload", type="filepath")
output_text = gr.Textbox(label="Meeting Summary")

app = gr.Interface(
    fn=transcript_audio,
    inputs=audio_input,
    outputs=output_text,
    title="AI Meeting Companion",
    description="Upload a meeting audio file to generate a summary of the key points."
)

app.launch(server_name="0.0.0.0", server_port=7860)
```

---

## Skills Demonstrated

This project demonstrates practical experience with:

* Building AI applications with Python
* Applying automatic speech recognition models
* Using transformer-based models from Hugging Face
* Integrating LLMs into an application workflow
* Prompt engineering for summarization
* Building interactive machine learning demos with Gradio
* Connecting speech-to-text systems with generative AI models
* Designing an end-to-end AI workflow from input to user-facing output

---

## Limitations

* The current version uses Whisper Tiny, which is lightweight but less accurate than larger Whisper models.
* The app does not yet identify individual speakers.
* The system currently returns summaries but does not separately extract action items, owners, or deadlines.
* Performance depends on the quality of the uploaded audio.
* Very long recordings may require additional optimization for chunking, memory usage, and processing time.

---

## Future Improvements

Planned improvements include:

* Add speaker diarization to identify different meeting participants
* Extract action items, deadlines, and responsible persons
* Add support for multilingual transcription
* Export summaries as PDF, Markdown, or text files
* Store meeting history in a database
* Add Docker support for easier deployment
* Deploy the application on Hugging Face Spaces or a cloud platform
* Compare Whisper Tiny with larger Whisper models for transcription accuracy

---

## Portfolio Context

This project was completed as part of my applied learning in generative AI and speech-to-text systems. I adapted the lab into a portfolio-ready implementation to demonstrate how automatic speech recognition and large language models can be combined to solve a real business productivity problem.

---

## Author

**Raphael Farodoye**
Data Scientist | Machine Learning | Generative AI | Python
