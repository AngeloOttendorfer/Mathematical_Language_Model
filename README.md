# Mathematical Language Model
This project includes an AI-Based-Language Model which is fine-tuned on mathematical tasks such as Algebra, Calculus and Geometry  to generate answers to corresponding mathematical problems.
The training and test data that is used contains LaTeX expressions which is an approach to inform the model about upcoming mathematical expressions.
<img width="1004" alt="Bildschirmfoto 2024-01-31 um 12 20 11" src="https://github.com/AngeloOttendorfer/Mathematical_Language_Model/assets/101193795/87a0eb2f-6ba0-4329-ab3b-fe7191f8559e">

## Pretrained Language Models
The first step is to compare the performance of different pretrained language models (PLM) on solving various categorical math questions.
Transformers from Huggingface is used as an orientation in selecting the PLMs. The following models are examined in this project:
### GPT
GPT employs the Transformer architecture, which utilizes self-attention mechanisms to weigh the importance of different words in a sentence. 
This allows the model to capture long-range dependencies and context effectively.
The model, once pre-trained, can then be fine-tuned for specific tasks. Given a prompt or input sequence, GPT uses its learned knowledge to generate coherent and contextually relevant text. 
The output is not just based on static patterns but adapts dynamically to the given context.

### BERT
BERT is pre-trained on large corpora using a masked language model objective. It learns to predict missing words in a sentence by considering both the left and right context. 
This bidirectional approach enables the model to grasp the full context of a word, making it highly effective in understanding nuances and dependencies within sentences.
Similar to GPT, BERT is built upon the Transformer architecture. This architecture utilizes self-attention mechanisms to weigh the importance of different words in a sentence. 
BERT employs multiple layers of transformers, allowing it to capture hierarchical representations of language.
BERT's pre<img width="1346" alt="Bildschirmfoto 2024-03-19 um 06 36 49" src="https://github.com/AngeloOttendorfer/Mathematical_Language_Model/assets/101193795/a43a9cab-10fb-4e5c-adae-1b2d51b59aad">
-training involves two key objectives - masked language model (MLM) and next sentence prediction (NSP). 
MLM tasks require the model to predict masked-out words in a sentence, fostering a deep understanding of word relationships.
NSP tasks involve predicting whether a pair of sentences is consecutive or not, promoting contextual understanding at a larger scale.

### T5
Text-to-Text Formulation: T5 frames all NLP tasks as converting input text to output text.
T5 builds on the Transformer architecture, which uses self-attention mechanisms to capture contextual relationships within sequences. 
T5 employs a stack of transformer layers to process and understand the input-output pairs effectively.
During pre-training, T5 uses a denoising autoencoder objective. It corrupts input-output pairs by masking some parts and trains the model to reconstruct the original text. 
This process helps the model learn robust representations and contextual understanding.

## Enhancement of mathematical language models
The second step will be to discover different approaches to enhance mathematical answer generation as the derivation of mathematical expression for models which is the big challenge because
math problem oriented answer generation mostly remains very low on performance. 

## Block diagram
![Block_diagramm_Mathematical_Language_Model](https://github.com/AngeloOttendorfer/Mathematical_Language_Model/assets/101193795/ef16f13d-3ef5-4d5a-a4b3-5189ccf657ed)

## Execution
* For the training (gpt2) run "python tune_model.py --dataroot 'DIRECTORY-TO-TRAININGDATA' --batch-size-per-replica BATCH-SIZE" in the command line
* For the training (bert) run "python tune_model.py --dataroot 'DIRECTORY-TO-TRAININGDATA' --batch-size-per-replica BATCH-SIZE --arch 'bert-base-uncased'" in the command line
* For the evaluation (gpt2) run "python eval_math_gpt.py --mathematica_dataroot 'DIRECTORY-TO-TESTDATA' --load 'DIRECTORY-TO-THE-CHECKPOINT-WITH-ALL-THE-RELEVANT-FILES'" in the      command line
* For the evaluation (gpt2) run "python eval_math_gpt.py --mathematica_dataroot 'DIRECTORY-TO-TESTDATA' --load 'DIRECTORY-TO-THE-CHECKPOINT-WITH-ALL-THE-RELEVANT-FILES' --arch 'bert-base-uncased'" in the command line

Die Batch-Size für die Evaluierung wird im Code angegeben (im DataLoader).

<img width="1344" alt="Bildschirmfoto 2024-03-19 um 06 36 09" src="https://github.com/AngeloOttendorfer/Mathematical_Language_Model/assets/101193795/f129267d-089e-45a0-a10f-4bb44dab0eb1">

<img width="1346" alt="Bildschirmfoto 2024-03-19 um 06 36 49" src="https://github.com/AngeloOttendorfer/Mathematical_Language_Model/assets/101193795/f92b280b-0ad4-4c96-8e92-a080df4562ba">
