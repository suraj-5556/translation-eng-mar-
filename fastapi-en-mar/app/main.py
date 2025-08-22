from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import pickle
# ------------------ FastAPI setup ------------------
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# ------------------ Load Models ------------------

# 1. Encoder-Decoder model (your custom model)
with open("https://github.com/suraj-5556/translation-eng-mar-/releases/download/v1.0/en-dec._model.h5", "rb") as f:
    model_ed = pickle.load(f)
# model_ed = load_model("(en-dec)_model.h5")
with open("app\models\english.pkl", "rb") as f:
    tokenizer_eng = pickle.load(f)
with open("app\models\marathi.pkl", "rb") as f:
    tokenizer_mar = pickle.load(f)

max_enc_len = 47   # set this to your encoder max length
max_dec_len = 35   # set this to your decoder max length

def translate_sentence_ed(input_text):
    enc_seq = tokenizer_eng.texts_to_sequences([input_text])
    enc_seq = pad_sequences(enc_seq, maxlen=max_enc_len, padding="post")

    start_id = tokenizer_mar.texts_to_sequences([["start"]])[0][0]
    end_id = tokenizer_mar.texts_to_sequences([["end"]])[0][0]

    dec_in = [start_id]
    out_words = []

    for _ in range(max_dec_len):
        dec_seq = pad_sequences([dec_in], maxlen=max_dec_len, padding="post")
        preds = model_ed.predict([enc_seq, dec_seq], verbose=0)
        pred_id = np.argmax(preds[0])

        if pred_id == end_id:
            break
        if pred_id not in (0, start_id, end_id):
            out_words.append(tokenizer_mar.index_word.get(pred_id, ""))

        dec_in.append(pred_id)

    return " ".join(out_words)

# 2. Transformer (your fine-tuned model on Hugging Face Hub)
transformer_model = AutoModelForSeq2SeqLM.from_pretrained("suraj5556/transformer-en-mar")
transformer_tokenizer = AutoTokenizer.from_pretrained("suraj5556/tokenizer-en-mar")
translator_ft = pipeline(
    "translation",
    model=transformer_model,
    tokenizer=transformer_tokenizer,
    src_lang="en_XX",
    tgt_lang="mr_IN"
)

# 3. Pretrained transformer model (Hugging Face official mBART)
translator_pretrained = pipeline(
    "translation",
    model="facebook/mbart-large-50-many-to-many-mmt",
    tokenizer="facebook/mbart-large-50-many-to-many-mmt",
    src_lang="en_XX",
    tgt_lang="mr_IN"
)

# ------------------ Routes ------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": ""})

@app.post("/translate", response_class=HTMLResponse)
async def translate(request: Request,
                    text: str = Form(...),
                    model_choice: str = Form(...)):
    
    if model_choice == "ed":
        result = translate_sentence_ed(text)
    elif model_choice == "ft":
        result = translator_ft(text, max_length=100, min_length=1, do_sample=False)[0]['translation_text']
    else:
        result = translator_pretrained(text, max_length=100, min_length=1, do_sample=False)[0]['translation_text']

    return templates.TemplateResponse("index.html", {
        "request": request,
        "text": text,
        "result": result,
        "model_choice": model_choice
    })

