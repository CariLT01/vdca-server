# VDCA Server

System for automating games on VDC.

## About

It will complete questions in practice mode automatically. You must have the game window open on one of your monitors and the program will take control of your mouse and keyboard. The server side is the one that is controlling the mouse and keyboard and the client side is the one collecting data about the current game using JavaScript.

The accuracy of the models may vary greatly between word to word. On average, the model has an average accuracy of 75%. However, it may vary greatly based on the words given.
The more the program gets familiar with the list, the higher accuracy it will be able to achieve (up to consistent 100%). By default, the program will keep
replaying the list until it assumes that 97% of the question space has been explored.

> [!WARNING]
> This program may trigger anti-robot defensive measures.
  

**Press the ESCAPE key to activate the kill switch!**
Press ESCAPE again to restart the application (might require a page refresh)

## How it works

### Multiple Choice:

**Reputation-based answer system**
A core characteristic of these games is that the same answer can be present across many questions. Using SQLite and a reputation-based system, the program is able to achieve higher scores by remembering which answers could go with which words. This alone is able to achieve accuracies of 75%+.

**Exact hash-based system**
For each question, a unique hash is computed for it. When the same question appears again, the answer is already known and can be used to answer directly without any heavy computation. Exact hash-based is counted towards the reputation score. If any reputation score is present, no further ML-based processing is done to save on compute costs and improve speeds. This achieves perfect accuracy when the same question is shown again.

**Embeddings and cosine similarities**
Using open-source embedding models, the program will try to predict which answer is the most likely based on how similar the meanings are. It will evaluate its confidence based on the confidence returned by the model and the difference between the confidence of other terms (evaluate ambuigity). If confidence is considered low, an LLM is used as a fallback.

**Large Language Model Processing**
LLMs are used as the last and final fallback if all the previous methods fail. It will try two providers: OpenRouter and GPT4Free. OpenRouter is the preferred route but if quota is exceeded, it will try GPT4Free which is less reliable. If LLM fails, it will fallback to using the results obtained via embeddings and cosine similarities. LLMs are extremely accurate but are very slow. To compute which answer the LLM actually chose for smaller LLMs that are bad at following instructions, we use `textdistance`.

**Select the most fitting image**

The model will guess randomly. It will remember the question and its answer to answer it correctly when reviewed again.

**Spelling**

The game conveniently exposed the answer in the document tree (DOM) of the page, it is just hidden and invisible to the average user. The client simply reads the answer from the DOM and inputs it into the box.
This guarantees that the model gets the correct answer every time.

**Natural Mouse Movements**
The program uses Bézier curves along with random points placed along the path to simulate a somewhat normal-looking human-like mouse path.

## Features:

- Remember previous questions that the bot failed
- Natural human-like behavior using randomization and bezier curves
- Semantic matching artificial intelligence model to predict the correct answer
- Ability to do spelling questions
- Fill in the blank artificial intelligence model for fill in the blank questions
- **Fully automatic with no human intervention**
- Use of LLM to answer questions via OpenRouter, extremely high accuracy. Only used if the semantic matching model is not confident enough or has high ambiguity between two words
- Uses an "answer reputation" system to score answers that have appeared in previous questions but not exactly the same question
- Scan question space with heuristic probability as a stopping point
- Lazy-loaded models to conserve memory
- Additionally, it includes protobuf-based serialization to serialize the full question space into a `.vcl` file, viewable at [https://vclviewer.web.app](https://vclviewer.web.app).

## System requirements

- At least **2 GB** of RAM
- Must be running Windows

It is possible that you must be **root** to run this program (for global hotkeys).

## Setup

1. Install Python 3.13
2. Create a virtual envrionment if necessary
3. Install everything in requirements.txt using `pip install -r requirements.txt`
4. Run the script for the first time (it may take a very long time until it has finished loading)
5. Pray that it works

Follow the client-side instructions if you haven't done so yet.
