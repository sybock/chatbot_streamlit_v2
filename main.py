import pandas as pd
import streamlit as st
import random
import torch

from streamlit.server.server import Server
import SessionState
import torch
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
#from pytorch_lightning.core.lightning import LightningModule

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.set_page_config(page_title="거침없이 하이킥 이순재 Bot") #layout='wide', initial_sidebar_state='auto'

# set first td and first th of every table to not display
st.markdown("""
<style>
table td:nth-child(1) {
    display: none
}
table th:nth-child(1) {
    display: none
}
</style>
""", unsafe_allow_html=True)

U_TKN = '<usr>'
S_TKN = '<sys>'
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'

def display_app_header(main_txt,sub_txt,is_sidebar = False):
    """
    function to display major headers at user interface
    Parameters
    ----------
    main_txt: str -> the major text to be displayed
    sub_txt: str -> the minor text to be displayed 
    is_sidebar: bool -> check if its side panel or major panel
    """

    html_temp = f"""
    <div style = "background.color:white  ; padding:15px">
    <h2 style = "color:#3c403f; text_align:center;"> {main_txt} </h2>
    <p style = "color:#3c403f; text_align:center;"> {sub_txt} </p>
    </div>
    """
    if is_sidebar:
        st.sidebar.markdown(html_temp, unsafe_allow_html = True)
    else: 
        st.markdown(html_temp, unsafe_allow_html = True)

def display_side_panel_header(txt):
    """
    function to display minor headers at side panel
    Parameters
    ----------
    txt: str -> the text to be displayed
    """
    st.sidebar.markdown(f'## {txt}')

@st.cache(allow_output_mutation=True, max_entries=1) #ttl=1200,
def load_model():
    model = GPT2LMHeadModel.from_pretrained('sybk/highkick-soonjae-v2')
    return model

@st.cache(allow_output_mutation=True, max_entries=1) #ttl=1200,
def load_reverse_model():
    model = GPT2LMHeadModel.from_pretrained('sybk/hk_backward_v2')
    return model

@st.cache(allow_output_mutation=True, max_entries=1) #ttl=1200,
def load_tokenizer():
    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK) 
    return tokenizer

@st.cache()
def cacherando():
    rando =random.random()
    return rando

def get_session_state(rando):
    session_state = SessionState.get(sessionstep = 0, random_number=random.random(), chatinput='', 
                    bot_input_ids='', chat_history_ids='', chathistory=[], bothistory=[], temperature='',
                    topk='', topp='')
    return session_state

    
class KoGPT2Chat():
    def __init__(self, model, tokenizer, reverse_model):
        self.total_msg_lst = []
        self.kogpt2 = model
        self.tok = tokenizer
        self.reverse_model = reverse_model

    def append_message(self, input, speaker):
        if len(self.total_msg_lst) > 5:
            self.total_msg_lst = self.total_msg_lst[:4]
        if speaker == 'user':
            self.total_msg_lst.append(U_TKN + input + EOS)
        else:
            self.total_msg_lst.append(S_TKN + input + EOS)

    def generate(self, input):
        a = ''
        input_ids = torch.LongTensor(self.tok.encode(U_TKN + input + SENT + S_TKN + a)).unsqueeze(dim=0)
        output_ids = self.kogpt2.generate(input_ids=input_ids, do_sample=True, max_length=128, \
                                   top_k=20, top_p=0.75, num_return_sequences=20,\
                                  no_repeat_ngram_size=2)
        output_lst = [self.tok.decode(out[input_ids.shape[-1]:], skip_special_tokens=True) for out in output_ids]
        output_lst = [o for o in output_lst if len(o) > 5]
        return output_lst
    
    def reranking(self, output_text: list, last_response: str):
        last_tokens = self.tok.encode(last_response, return_tensors='pt')
        loss_list = []
        kor_list = ["가", "요","이","에", "예", "네", "어", "세", " ", "셔", "셨", ".",",","?","!","야","죠","하", "고"]
        last_response_keywords = [c for c in last_response if c not in kor_list]
        for o in output_text:
            output_token = self.tok.encode(o.strip(), return_tensors='pt')
            inputs = torch.cat((output_token, last_tokens), dim=1)
            mask = torch.full_like(output_token, -100, dtype=torch.long)
            labels = torch.cat((mask, last_tokens), dim=1)
            reverse_outputs = self.reverse_model(inputs, labels=labels)
            #reverse_outputs = REVERSE_MODEL(inputs)
            loss = (reverse_outputs.loss).float()
            for char in o:
                if char in last_response_keywords: loss += 1.5
            loss_list.append(-loss.float())
        best_output = output_text[loss_list.index(min(loss_list))]
        return best_output

    def fixed_messages(self, input_msg: str):
        fixed_msg_list = ["이름이 뭐예요", "이름이 뭐예요?", "직업이 뭐예요", "직업이 뭐예요?", "몇 살이에요?"]
        output=[]
        if input_msg in fixed_msg_list:
            msg_id = fixed_msg_list.index(input_msg)
            if msg_id == 0 or msg_id == 1: output = random.choice(["이순재. 왜그래?", '이순재. 너 임마.'])
            if msg_id == 2 or msg_id == 3: output = random.choice(["이&박 여성전문 한방병원 원장이다 임마.", "한의사... 이 자식이... 왜 이래?"])
            if msg_id == 4: output = random.choice(["야 임마... 넌 몇 살이야!?", "이 자식이... 왜 이래? 35년생이다 왜!"])

        return output

    def mychat(self):
        rando = cacherando()
        session_state = get_session_state(rando)

        display_side_panel_header("Configuration & Fine-Tuning")

        #session_state.temperature = st.sidebar.slider("Choose temperature. Higher means more creative (crazier): ", 0.0, 3.0, 0.8, 0.1)
        session_state.topk = st.sidebar.slider("Choose Top K, the number of words considered at each step. Higher is more diverse; 0 means infinite:", 0, 200, 20)
        session_state.topp = st.sidebar.slider("Choose Top P. Limits next word choice to higher probability; lower allows more flexibility:", 0.0, 1.0, 0.70, 0.05)
        session_state.no_repeat_ngram_size = st.sidebar.slider("No Repeat N-Gram Size. Eliminates repeated phrases of N Length", 0, 6, 3)
        generation_config = {"topk":session_state.topk, "topp":session_state.topp, "no_repeat_ngram_size":session_state.no_repeat_ngram_size}


        chatlogholder = st.empty()
        chatholder = st.empty()

        q = chatholder.text_input(">> You:", value="").strip()
        
        try:
            if st.button("Chat"):
            #if len(q) > 1:
                session_state.sessionstep += 1
                self.append_message(q, speaker='user')
                a = self.fixed_messages(q)
                if len(a) == 0:
                    a_list = self.generate(q)
                    a_list = [a for a in a_list if '박' not in a and '순' not in a]
                    #print(a_list)
                    a = self.reranking(a_list, q)
                self.append_message(a, speaker='bot')

                session_state.chathistory.append(q)
                session_state.bothistory.append(a)

                df = pd.DataFrame(
                    {
                    'You': session_state.chathistory,
                    'Bot': session_state.bothistory
                    })

                with chatlogholder:
                    st.table(df.tail(5))
                with st.beta_expander("Full chat history"):
                    st.table(df)       
        except:
            raise


def main():
    main_txt = """🤬🗣🤪 거침없이 하이킥 이순재 BOT 🤬🗣🤪"""
    sub_txt = "MBC 2006.11.06. ~ 2007.07.13."
    subtitle = """
        <p style="text-align:center">이순재 Bot은 SKT에서 공개한 kogpt2 v2 기반으로 개발된 챗봇입니다.<br><br>
        구어 대화체를 익히기 위해 먼저 한국어 Conversation Data로 학습하였고,<br> '이순재' 캐릭터를 모방하기 위해 거침없이 하이킥 대본으로 fine-tuning을 진행했습니다.<br>
        더 다채로운 챗봇 답변을 위해 DialoGPT (Zhang et al., 2019) 모델과 같은 방식으로 답변을 Re-Ranking하는 시스템을 구축했습니다. 
        또한, 자연스러운 답변을 유도하기 위해서 Scoring System을 개선하였고, 자동 응답을 출력하는 질문을 추가했습니다. <br><br>
        국민 시트콤의 국민 할아버지, 거침없이 하이킥의 이순재와 대화를 해보세요!😁</p>
        """

    display_app_header(main_txt,sub_txt,is_sidebar = False)
    st.image("https://i.ibb.co/8dxbtn5/sj-2.png", width=650)
    st.markdown(f"<div style='text-align: justify;'> {subtitle} </div>", unsafe_allow_html = True)
    display_side_panel_header("Resources")
    st.sidebar.markdown("""
                        [Github](https://github.com/sybock)  
                        """)
    with st.spinner("Initial models loading, please be patient"):
        tokenizer = load_tokenizer() 
        model = load_model()
        reverse_model = load_reverse_model()
    chatbot = KoGPT2Chat(model, tokenizer, reverse_model)
    chatbot.mychat()
    st.markdown("""
---
### About
- Trained using [kogpt2 v2](https://github.com/SKT-AI/KoGPT2) released by SKT
- Training data: Sejong Spoken Corpus (8MB), [Conversation Data (1MB)](https://github.com/haven-jeon/KoGPT2-chatbot), Highkick Script (5MB)
- Streamlit chatbot format was adapted from [here](https://github.com/bigjoedata/jekyllhydebot)
""")

if __name__ == "__main__":
    main()
