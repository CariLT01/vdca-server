# VDCA Server

System for automating games on VDC.

## About

It will complete questions in practice mode automatically. You must have the game window open on one of your monitors and the program will take control of your mouse and keyboard. The server side is the one that is controlling the mouse and keyboard and the client side is the one collecting data about the current game using JavaScript.

The accuracy of the models may vary greatly between word to word. On average, the model has an average accuracy of 75%. However, it may vary greatly based on the words given.

> [!WARNING]
> This program may trigger anti-robot defensive measures.
  

**Press the ESCAPE key to activate the kill switch!**
You must terminate and restart the VDCA server after you have activated the kill switch.

## How it works

**Multiple choice questions**

Uses the *thenlper/gte-base* model on Hugging Face to select the best answer based on similarity between the target word and the possible answers. The contextual sentence provided is ignored to improve model performance and also because in the game, all possible answers have very different meanings and therefore have very different vectors. Therefore, the context in which the word is in will not have such a signficant impact on accuracy.

It will guess based on semantic meaning when it first encounters this question. If the bot answered incorrectly, it will remember the question and its answer to answer it correctly when reviewed.

**Fill in the blank**

Uses the Bart model (by Facebook) to predict what word best fits in the omitted part of the sentence. However, the approach sucks so words in the vocabulary list are prioritized by multiplying the probability outputed by the model by a coefficient.

Like the multiple choice questions, it will guess based on semantic meaning when it first encounters this question. If the bot answered incorrectly, it will remember the question and its answer to answer it correctly when reviewed.

**Select the most fitting image**

The model will guess randomly. It will remember the question and its answer to answer it correctly when reviewed again.

**Spelling**

The game conveniently exposed the answer in the document tree (DOM) of the page, it is just hidden and invisible to the average user. The client simply reads the answer from the DOM and inputs it into the box.
This guarantees that the model gets the correct answer every time.



## Features:

- Remember previous questions that the bot failed
- Natural human-like behavior using randomization and bezier curves
- Semantic matching artificial intelligence model to predict the correct answer
- Ability to do spelling questions
- Fill in the blank artificial intelligence model for fill in the blank questions
- **Fully automatic with no human intervention**
- Use of LLM to answer questions via OpenRouter, extremely high accuracy. Only used if the semantic matching model is not confident enough or has high ambiguity between two words
- Uses an "answer reputation" system to score answers that have appeared in previous questions but not exactly the same question

## System requirements

- At least **8 GB** of RAM
- Must be running Windows

It is possible that you must be **root** to run this program.

## Setup

1. Install Python 3.13
2. Create a virtual envrionment if necessary
3. Install everything in requirements.txt using `pip install -r requirements.txt`
4. Run the script for the first time (it may take a very long time until it has finished loading)
5. Pray that it works

Follow the client-side instructions if you haven't done so yet.
