
import re
import streamlit as st
import torch
import joblib
import pickle

from transformers import (
    pipeline,
    BertTokenizer,
    BertForSequenceClassification,
    BartTokenizer,
    BartForConditionalGeneration,
    BertForTokenClassification,
    BertTokenizerFast,
)

# ============================================
# PATHS
# ============================================

# ---- Classification paths
CLS_MODEL_PATH = "./Classification/Bert"
LABEL_ENCODER_PATH = "./Classification/label_encoder.pkl"

# ---- Summarization paths
SUM_MODEL_PATH = "./Summarization/bart_Summary_finetuned"

# ---- NER paths
NER_TOKENIZER_PATH = "./NER/bert_ner_model"
NER_MODEL_PATH = "./NER/bert_ner_model"
ID2TAG_PATH = "./NER/Ner_Models/id2tag.pkl"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# ======================================================
# =========== Classification Helper ====================
# ======================================================

def clean_text_strong(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+|pic\.twitter\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[\(\[].*?[\)\]]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r'[^A-Za-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ======================================================
# MODEL LOADERS
# ======================================================

@st.cache_resource
def load_classifier():
    tokenizer = BertTokenizer.from_pretrained(CLS_MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(CLS_MODEL_PATH)
    model.to(device)
    model.eval()
    le = joblib.load(LABEL_ENCODER_PATH)
    return tokenizer, model, le


@st.cache_resource
def load_summarizer():
    tokenizer = BartTokenizer.from_pretrained(SUM_MODEL_PATH)
    model = BartForConditionalGeneration.from_pretrained(SUM_MODEL_PATH)

    device_idx = 0 if torch.cuda.is_available() else -1
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=device_idx)

    return summarizer, tokenizer


# ======================================================
# ========= NER HELPER FUNCTIONS =======================
# ======================================================

@st.cache_resource
def load_ner_model():
    # tokenizer = BertTokenizer.from_pretrained(NER_TOKENIZER_PATH)
    
    tokenizer = BertTokenizerFast.from_pretrained(NER_TOKENIZER_PATH)

    
    model = BertForTokenClassification.from_pretrained(
        NER_MODEL_PATH
    )

    model.to(device)
    model.eval()

    with open(ID2TAG_PATH, "rb") as f:
        id2tag = pickle.load(f)

    return tokenizer, model, id2tag



def run_ner(text, tokenizer, model, id2tag):

    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    input_ids = encoding["input_ids"][0]
    offsets = encoding["offset_mapping"][0].tolist()

    input_ids = input_ids.unsqueeze(0).to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask).logits

    predicted_ids = logits[0].argmax(-1).tolist()

    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])

    entities = {}
    current_label = None
    current_tokens = []

    for token, pred_id, (start, end) in zip(tokens, predicted_ids, offsets):
        tag = id2tag[pred_id]

        if tag == "O":
            if current_tokens:
                entity = " ".join(current_tokens).strip()
                if len(entity) > 1:
                    entities.setdefault(current_label, []).append(entity)
            current_label = None
            current_tokens = []
            continue

        split = tag.split("-")
        if len(split) != 2:
            continue

        prefix, label = split

        word = text[start:end].strip()

        if prefix == "B":
            if current_tokens:
                entity = " ".join(current_tokens).strip()
                if len(entity) > 1:
                    entities.setdefault(current_label, []).append(entity)

            current_label = label
            current_tokens = [word]

        elif prefix == "I" and current_label == label:
            current_tokens.append(word)
        else:
            # Forced new entity
            if current_tokens:
                entity = " ".join(current_tokens).strip()
                if len(entity) > 1:
                    entities.setdefault(current_label, []).append(entity)
            current_label = None
            current_tokens = []

    # finalize last entity
    if current_tokens:
        entity = " ".join(current_tokens).strip()
        if len(entity) > 1:
            entities.setdefault(current_label, []).append(entity)

    # remove duplicates and single-letter noise
    for k in entities:
        unique = []
        for v in entities[k]:
            if len(v) > 1 and v not in unique:
                unique.append(v)
        entities[k] = unique

    return entities





# Load models
cls_tokenizer, cls_model, le = load_classifier()
summarizer, sum_tokenizer = load_summarizer()
ner_tokenizer, ner_model, id2tag = load_ner_model()







## Streamlit UI Part


st.set_page_config(
    page_title="📰 News AI Suite",
    page_icon="🧠",
    layout="wide",
)

# Custom Styling
st.markdown("""
<style>
    .stTextArea textarea {
        font-size: 15px !important;
        line-height: 1.5 !important;
    }
    .main-title {
        font-weight: 900;
        font-size: 36px;
        color: #1c4e80;
        text-align:center;
        padding-bottom:12px;
    }
    .sub-head {
        color: #5383c3;
        font-weight:600;
        font-size: 20px;
    }
    .result-box {
        padding: 15px;
        background: #1e1e1e !important;
        border: 1px solid #444 !important;
        border-radius: 12px;
        color: white !important;
    }
    .result-box h4, .result-box p {
        color: white !important;
    }

</style>
""", unsafe_allow_html=True)

st.markdown("<h2 class='main-title'>📰 AI-Powered News Analyzer</h2>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "📊 Category Classification",
    "📝 Smart Summarization",
    "🧬 Named Entity Recognition"
])

# ============================================================== TAB-1 CLASSIFICATION
with tab1:
    st.markdown("<p class='sub-head'>Predict your news article category</p>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        text = st.text_area("Enter news text below:", height=230)

    with col2:
        st.info("""
        📌 This model predicts categories like:
        - Politics  
        - Sports  
        - Technology  
        - Entertainment  
        - Business  
        """)
        st.warning("🚨 Ensure meaningful text for accurate results.")

    if st.button("🔍 Classify Article"):
        if not text.strip():
            st.warning("Enter text first")
        else:
            inputs = cls_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=96
            ).to(device)

            with torch.no_grad():
                outputs = cls_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)

            idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0][idx].item()
            label = le.inverse_transform([idx])[0]

            st.markdown("### 🎯 Classification Result")
            st.markdown(f"""
            <div class="result-box">
                <h4>🏷 Predicted Category: <b>{label}</b></h4>
                <p>🔥 Confidence: <b>{confidence*100:.2f}%</b></p>
            </div>
            """, unsafe_allow_html=True)



# ============================================================== TAB-2 SUMMARIZATION
with tab2:
    st.markdown("<p class='sub-head'>Generate summary with bullet points</p>", unsafe_allow_html=True)

    article = st.text_area("Paste article", height=250)

    if st.button("📝 Generate Summary"):
        if not article.strip():
            st.warning("Paste article")
        else:
            with st.spinner("⏳ Creating summary..."):

                encoded = sum_tokenizer(
                    article,
                    truncation=True,
                    max_length=1024,
                    return_tensors="pt"
                )

                safe_text = sum_tokenizer.decode(
                    encoded["input_ids"][0],
                    skip_special_tokens=True
                )

                summary = summarizer(
                    safe_text,
                    max_length=200,
                    min_length=60,
                    num_beams=5,
                    repetition_penalty=2.0,
                    no_repeat_ngram_size=4
                )

                summary_txt = summary[0]["summary_text"].strip()

                import re
                sentences = re.split(r"(?<=[.!?])\s+", summary_txt)
                bullets = "\n".join([f"✔️ {s}" for s in sentences if s.strip()])

                st.success("Summary is ready")

                st.markdown("### 📌 Summary Overview")
                final_html = bullets.replace('\n', '<br>')

                st.markdown(f"""
                <div class="result-box">
                    {final_html}
                </div>
                """, unsafe_allow_html=True)




# ============================================================== TAB-3 NER
with tab3:
    st.markdown("<p class='sub-head'>Extract entities like PER, ORG, LOC</p>", unsafe_allow_html=True)

    ner_text = st.text_area("Enter article for NER", height=200)

    if st.button("🧬 Extract Entities"):
        if not ner_text.strip():
            st.warning("Enter text first")
        else:
            with st.spinner("🔍 Identifying entities..."):
                entities = run_ner(ner_text, ner_tokenizer, ner_model, id2tag)

            st.success("✔ Entities Extracted")

            if not entities:
                st.info("No named entities found")
            else:
                st.markdown("### 🧠 Extracted Entity Table")
                rows = [{"Entity Type": k, "Value": v} for k in entities for v in entities[k]]
                st.dataframe(rows, use_container_width=True)

                st.markdown("### 📌 Categorized Entities")
                for k, v in entities.items():
                    st.markdown(f"#### 🏷 {k.upper()}")
                    st.write(", ".join(sorted(set(v))))

