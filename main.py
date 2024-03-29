import streamlit as st
import openai

# OpenAI API 키 설정
openai.api_key = "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"

# GPT-3 대화 모델 설정
model = "text-davinci-002"

# Streamlit 앱의 제목 설정
st.title("Chat with GPT-3")

# 사용자 입력 받기
user_input = st.text_input("You:", "")

if user_input:
    # OpenAI GPT-3를 사용하여 대화 생성
    response = openai.Completion.create(
        engine=model,
        prompt=user_input,
        max_tokens=50
    )
    
    # 생성된 대화 출력
    st.text("GPT-3: " + response.choices[0].text.strip())
