# 거침없이 하이킥 이순재 Bot

이순재 Bot은 SKT에서 공개한 kogpt2 v2 기반으로 개발된 챗봇입니다.<br><br>
구어 대화체를 익히기 위해 먼저 한국어 Conversation Data로 학습하였고,'이순재' 캐릭터를 모방하기 위해 거침없이 하이킥 대본으로 fine-tuning을 진행했습니다.<br>
더 다채로운 챗봇 답변을 위해 DialoGPT (Zhang et al., 2019) 모델과 같은 방식으로 답변을 Re-Ranking하는 시스템을 구축했습니다. 
또한, 자연스러운 답변을 유도하기 위해서 Scoring System을 개선하였고, 자동 응답을 출력하는 질문을 추가했습니다. <br><br>
국민 시트콤의 국민 할아버지, 거침없이 하아킥의 이순재와 대화를 해보세요!😁

[Demo](https://share.streamlit.io/sybock/chatbot_streamlit_v2/main/main.py)

### About
- Trained using [kogpt2 v2](https://github.com/SKT-AI/KoGPT2) released by SKT
- Training data: Sejong Spoken Corpus (8MB), [Conversation Data (1MB)](https://github.com/haven-jeon/KoGPT2-chatbot), Highkick Script (5MB)
- Streamlit chatbot format was adapted from [here](https://github.com/bigjoedata/jekyllhydebot)
