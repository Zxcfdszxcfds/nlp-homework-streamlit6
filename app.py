import streamlit as st
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import re

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

# ---------------------- 模块1：词义消歧（WSD，纯Python实现） ----------------------
def simple_wsd(sentence, word):
    """基于规则的简化词义消歧（模拟Lesk算法效果）"""
    if word == "bank":
        if "deposit" in sentence.lower() or "money" in sentence.lower():
            return "金融机构：A financial institution where people deposit or borrow money."
        elif "river" in sentence.lower() or "water" in sentence.lower():
            return "河岸：The sloping land alongside a body of water."
    elif word == "apple":
        if "company" in sentence.lower() or "smartphone" in sentence.lower():
            return "科技公司：A multinational technology company."
        else:
            return "水果：The round fruit of a tree of the rose family."
    else:
        return "未找到对应词义"

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

# ---------------------- 模块2：语义角色标注（SRL，纯Python实现） ----------------------
def simple_srl(text):
    """基于规则的轻量级语义角色标注"""
    tokens = re.findall(r'\b\w+\b', text)
    results = []
    
    # 1. 找到动词作为谓词
    verbs = ["is", "are", "was", "were", "manufacturing", "making", "producing", "selling"]
    predicates = []
    for i, token in enumerate(tokens):
        if token.lower() in verbs:
            predicates.append((token, i))
    
    for pred_word, pred_idx in predicates:
        args = {"Predicate": pred_word}
        
        # 2. 找主语（位于动词前的专有名词）
        for i in range(pred_idx):
            word = tokens[i]
            if word[0].isupper() and word not in ["The", "A", "An"]:
                args["A0 (Agent)"] = word
                break
        
        # 3. 找宾语（位于动词后的名词）
        for i in range(pred_idx + 1, len(tokens)):
            word = tokens[i]
            if word not in ["the", "a", "an", "in", "at", "on"]:
                args["A1 (Patient)"] = word
                break
        
        # 4. 找地点/时间修饰语（简单介词短语识别）
        prep_words = ["in", "at", "on", "for"]
        for i in range(len(tokens)):
            word = tokens[i]
            if word in prep_words and i + 1 < len(tokens):
                next_word = tokens[i+1]
                if word in ["in", "at", "on"]:
                    args["AM-LOC"] = next_word
                elif word in ["for", "by"]:
                    args["AM-TMP"] = next_word
                break
        
        results.append(args)
    return results

def get_srl_tree_text(text):
    """生成简单的依存关系文本"""
    tokens = re.findall(r'\b\w+\b', text)
    return "\n".join([f"{word}" for word in tokens])

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
    st.markdown("对比传统规则方法与BERT上下文向量的消歧效果")
    
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
            # 规则方法
            def1 = simple_wsd(sent1, target_word)
            def2 = simple_wsd(sent2, target_word)
            
            # BERT向量
            emb1 = get_bert_embedding(sent1, target_word, tokenizer, bert_model)
            emb2 = get_bert_embedding(sent2, target_word, tokenizer, bert_model)
            
            st.subheader("规则方法结果")
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
    st.markdown("基于规则的轻量级语义角色标注，提取谓词与论元")
    
    input_text = st.text_input("输入句子", value="Apple is manufacturing new smartphones in China this year.")
    
    if st.button("执行SRL标注", key="srl_btn"):
        with st.spinner("分析中..."):
            results = simple_srl(input_text)
            
            st.subheader("结构化结果")
            if results:
                st.dataframe(pd.DataFrame(results))
            
            st.subheader("依存关系文本")
            st.text(get_srl_tree_text(input_text))

# ---------------------- 页脚 ----------------------
st.markdown("---")
st.markdown("© 2025 NLP 课程实验 | 语义分析综合平台")
