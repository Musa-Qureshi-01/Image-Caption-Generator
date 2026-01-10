<!-- Banner -->
<h1 align="center">Image Caption Generator (CNN + LSTM)</h1>

<p align="center">
  This repository contains a complete deep learning system that generates natural language captions from images using 
  <b>DenseNet201</b> for feature extraction and a <b>custom LSTM decoder</b>. The project includes a full training pipeline, 
  preprocessing, feature engineering, inference system, and an interactive <b>Streamlit web application</b>.
</p>

<p align="center">
  <img src="https://i.imgur.com/d8JvGhd.jpeg" width="750" style="border: 5px solid black; border-radius: 6px;">
</p>

---

<h2 id="content">Content</h2>

<ul>
  <li><a href="#image-caption-generator">Image Caption Generator</a>
    <ul>
      <li><a href="#content">Content</a></li>
      <li><a href="#description">Description</a></li>
      <li><a href="#dataset">Dataset</a></li>
      <li><a href="#features">Features</a></li>
      <li><a href="#install-prerequisites">Install Prerequisites</a></li>
      <li><a href="#notebook-explanation">Notebook Explanation</a></li>
      <li><a href="#model-architecture">Model Architecture</a></li>
      <li><a href="#training-pipeline">Training Pipeline</a></li>
      <li><a href="#inference-pipeline">Inference Pipeline</a></li>
      <li><a href="#streamlit-app">Streamlit App</a></li>
      <li><a href="#results">Results</a></li>
      <li><a href="#author">Author</a></li>
      <li><a href="#contributing">Contributing</a></li>
    </ul>
  </li>
</ul>

---

<h2 id="description">Description</h2>

This project generates meaningful captions for images by combining:

- A CNN encoder (DenseNet201)  
- A custom LSTM decoder  
- A tokenized vocabulary  
- A Seq2Seq-like architecture  

The system reads an image → extracts features → predicts caption tokens → forms a complete sentence.

This repository includes the entire pipeline from preprocessing to deployment.

---

<h2 id="dataset">Dataset</h2>

This project uses the Flickr8k Dataset, containing:

- Over 8,000 real-world images  
- Five human-written captions per image  
- Rich natural language descriptions  

Dataset link:  
https://www.kaggle.com/datasets/adityajn105/flickr8k

---

<h2 id="features">Features</h2>

- CNN + LSTM hybrid deep learning model  
- DenseNet201 feature extractor  
- Custom caption decoder  
- Streamlit-based Web App  
- Word-by-word caption generation  
- Full inference implementation  
- Internal working explanations  
- GPU-friendly  
- HuggingFace deploy-ready  

---

<h2 id="install-prerequisites">Install Prerequisites</h2>

Make sure Python 3.8+ is installed.

Install dependencies:

```bash
pip install -r requirements.txt
Main libraries:

TensorFlow

Keras

NumPy

Pandas

Matplotlib

Pillow

Streamlit
```
<h2 id="notebook-explanation">Notebook Explanation</h2>
The Jupyter Notebook covers:

1. Caption Preprocessing
Lowercasing

Cleaning

Tokenization

Vocabulary creation

Sequence padding

2. Feature Extraction
DenseNet201 (without classification layers)

1920-dimensional embeddings

Extract features for all images

3. Decoder Model
Embedding layer

LSTM for caption generation

Concatenation with CNN features

Softmax layer

4. Training
EarlyStopping

ReduceLROnPlateau

Saving the best model (.keras)

5. Inference
Predict token-by-token

Stop at "endseq."

Convert indices to words

<h2 id="model-architecture">Model Architecture</h2>
Encoder – DenseNet201
Extracts image features:

css
Copy code
Image → DenseNet201 → Feature Vector (1920 dimensions)
Decoder – LSTM caption generator
Takes the feature vector + caption prefix → predicts next word.

<p align="center"> <img src="https://i.ibb.co/4NpRZ0y/caption-architecture.png" width="850" style="border: 5px solid black; border-radius: 6px;"> </p>
<h2 id="training-pipeline">Training Pipeline</h2>
The model is trained by:

Preprocessing captions

Extracting DenseNet features

Tokenizing sentences

Creating sequence pairs

Training the decoder LSTM

Validating performance
Includes callbacks:

EarlyStopping

ReduceLROnPlateau

ModelCheckpoint

<h2 id="inference-pipeline">Inference Pipeline</h2>
A minimal caption generation loop:

python
Copy code
in_text = "startseq"

for _ in range(max_length):
    seq = tokenizer.texts_to_sequences([in_text])[0]
    seq = pad_sequences([seq], maxlen=max_length)

    y = caption_model.predict([features, seq])
    word = tokenizer.index_word.get(y.argmax())

    if word == "endseq":
        break

    in_text += " " + word

caption = in_text.replace("startseq", "").strip()
<h2 id="streamlit-app">Streamlit App</h2>
The app includes:

Drag-and-drop image upload

Image preview

Caption generation

Clean UI

Base64 image embedding

HuggingFace deployment support

Run locally:

bash
Copy code
streamlit run main.py

<h2 id="results">Results</h2>
Example output captions:

"A brown dog running through a field."
"A group of people sitting at a table outdoors."
"A child playing with a toy on the floor."

The model performs well on natural images, outdoor scenes, animals, and object-centric photos.

<h2 id="author">Author</h2>
Musa Qureshi
GitHub: https://github.com/Musa-Qureshi-01
LinkedIn: https://www.linkedin.com/in/musaqureshi
Twitter (X): https://x.com/Musa_Qureshi_01

<h2 id="contributing">Contributing</h2>
Pull requests are welcome.
For major changes, please open an issue to discuss what you would like to update.

Please include updates to tests where applicable.
