import streamlit as st
import nltk
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import spacy
from spacy import displacy

# ---------------------- 页面配置 ----------------------
st.set_page_config(
    page_title="语义分析综合平台",
    page_icon="📚",
    layout="wide"
)

# ---------------------- 预加载模型（轻量版） ----------------------
@st.cache_resource
def load_bert():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    return tokenizer, model

@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

# ---------------------- 模块1：词义消歧（WSD） ----------------------
def lesk_wsd(sentence, word):
    """Lesk算法词义消歧"""
    synset = lesk(sentence.split(), word)
    if synset:
        return synset, synset.definition()
    else:
        return None, "未找到对应词义"

def get_bert_embedding(sentence, word, tokenizer, model):
    """获取目标词的BERT上下文向量"""
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = model(**inputs)
    last_hidden = outputs.last_hidden_state
    
    # 找到目标词在token中的位置
    word_tokens = tokenizer.tokenize(word)
    sentence_tokens = tokenizer.tokenize(sentence.lower())
    word_idx = None
    for i in range(len(sentence_tokens) - len(word_tokens) + 1):
        if sentence_tokens[i:i+len(word_tokens)] == word_tokens:
            word_idx = i + 1  # +1 因为有[CLS]
            break
    if word_idx is None:
        return None
    return last_hidden[0, word_idx, :].detach().numpy()

def cosine_similarity(a, b):
    """计算余弦相似度"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ---------------------- 模块2：语义角色标注（SRL） ----------------------
def simple_srl(text, nlp):
    """基于spaCy的轻量级语义角色标注"""
    doc = nlp(text)
    srl_results = []
    for token in doc:
        if token.pos_ == "VERB":
            predicate = token.text
            args = {}
            for child in token.children:
                if child.dep_ == "nsubj":
                    args["A0 (Agent)"] = child.text
                elif child.dep_ == "dobj":
                    args["A1 (Patient)"] = child.text
                elif child.dep_ == "prep":
                    for grandchild in child.children:
                        if grandchild.dep_ == "pobj":
                            if child.text in ["in", "at", "on"]:
                                args["AM-LOC"] = grandchild.text
                            elif child.text in ["for", "by"]:
                                args["AM-TMP"] = grandchild.text
            srl_results.append({"Predicate": predicate, **args})
    return srl_results, doc

# ---------------------- 页面内容 ----------------------
st.title("📚 语义分析综合平台")
st.markdown("---")

tab1, tab2 = st.tabs([
    "模块1：词义消歧（WSD）对比测试",
    "模块2：语义角色标注（SRL）提取与可视化"
])

# ---------------------- 模块1：词义消歧 ----------------------
with tab1:
    st.header("🔤 词义消歧（WSD）对比测试")
    st.markdown("对比传统Lesk算法与BERT上下文向量的消歧效果")
    
    # 加载模型
    tokenizer, bert_model = load_bert()
    
    # 输入部分
    col1, col2 = st.columns(2)
    with col1:
        sent1 = st.text_input("句子1", value="I went to the bank to deposit my money.")
        target_word = st.text_input("目标多义词", value="bank")
    with col2:
        sent2 = st.text_input("句子2", value="I sat by the river bank.")
    
    if st.button("执行消歧与对比", key="wsd_btn"):
        with st.spinner("计算中..."):
            # Lesk算法
            synset1, def1 = lesk_wsd(sent1, target_word)
            synset2, def2 = lesk_wsd(sent2, target_word)
            
            # BERT向量
            emb1 = get_bert_embedding(sent1, target_word, tokenizer, bert_model)
            emb2 = get_bert_embedding(sent2, target_word, tokenizer, bert_model)
            
            st.subheader("Lesk算法结果")
            st.write(f"句子1词义: {def1}")
            st.write(f"句子2词义: {def2}")
            
            if emb1 is not None and emb2 is not None:
                sim = cosine_similarity(emb1, emb2)
                st.subheader("BERT上下文向量相似度")
                st.write(f"余弦相似度: {sim:.4f}")
                if sim < 0.7:
                    st.info("相似度较低，验证了BERT的动态上下文向量特性")

# ---------------------- 模块2：语义角色标注 ----------------------
with tab2:
    st.header("🎭 语义角色标注（SRL）提取与可视化")
    st.markdown("基于spaCy的轻量级语义角色标注，提取谓词与论元")
    
    # 加载模型
    nlp = load_spacy()
    
    input_text = st.text_input("输入句子", value="Apple is manufacturing new smartphones in China this year.")
    
    if st.button("执行SRL标注", key="srl_btn"):
        with st.spinner("分析中..."):
            results, doc = simple_srl(input_text, nlp)
            
            st.subheader("结构化结果")
            if results:
                st.dataframe(pd.DataFrame(results))
            
            st.subheader("依存关系可视化")
            # 生成displacy HTML
            html = displacy.render(doc, style="dep", jupyter=False)
            st.markdown(f'<div style="overflow-x: auto;">{html}</div>', unsafe_allow_html=True)

# ---------------------- 页脚 ----------------------
st.markdown("---")
st.markdown("© 2025 NLP 课程实验 | 语义分析综合平台")
