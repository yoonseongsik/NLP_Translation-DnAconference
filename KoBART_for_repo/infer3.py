import torch
import streamlit as st
from kobart import get_kobart_tokenizer
import transformers
from transformers.models.bart import BartForConditionalGeneration

@st.cache
def load_model():
    model = BartForConditionalGeneration.from_pretrained('/content/drive/Shareddrives/D&A/모델/KoBART/KoBART-translation-main/kobart_translation')
    tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1')
    return model

model = load_model()
tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1')
st.title("KoBART Translation Test")
text = st.text_area("한글 문장 입력:")

st.markdown("### 한글 문장")
st.write(text)

if text:
    text = text.replace('\n', '')
    st.markdown("### KoBART Translation 결과")
    with st.spinner('processing..'):
        input_ids = tokenizer.encode(text)
        input_ids = torch.tensor(input_ids)
        input_ids = input_ids.unsqueeze(0)
        output = model.generate(input_ids, eos_token_id=1, max_length=512, num_beams=5)
        output = tokenizer.decode(output[0], skip_special_tokens=True)
    st.write(output)