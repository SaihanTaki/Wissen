<div align="center">
<h1>Wissen</h1> 
<h4 style="font-style: italic">In  German, <code>Wissen</code> means <code>Knowledge</code></h4>
</div>



## 💡 About
Wissen is a  LLM-powered chatbot that can handle conversations across multiple documents.
This app is built with
- Langchain
- Google Palm
- Streamlit

**Wissen has two ways to chat with you:**

1. `Document Q&A Mode:`
- Uses Google's powerful Palm Base model for answering your questions.
- Activated when you upload documents to Wissen.
- In this mode, Wissen can only answer questions about the documents you've uploaded.

2. `Normal Conversation Mode:`
- This is the default mode when you start Wissen.
- Uses Google's Palm Chat model for open-ended conversation.
- Ask anything you want in this mode.
- Wissen also switches to this mode when you clear all uploaded documents.

**Switching between modes is automatic:**
- Uploading documents puts Wissen in Document Q&A mode.
- Clearing all documents brings Wissen back to Normal Conversation mode.

## 🛠️ Installation

First clone the repository
```
$ git clone https://github.com/SaihanTaki/Wissen.git
```
Change directory to wissen 
```
$ cd Wissen
```
Build the docker image 
```
$ docker build -t wissen -f docker/Dockerfile .
```
Run the container

```
$ docker run -p 8080:8501 -e GOOGLE_API_KEY=paste_your_google_api_key wissen
```

Go to `localhost:8080` in your browser 

**N.B.** you can find an api key here [Google AI Studio  API Key](https://aistudio.google.com/app/apikey)

## 🛡️ License
Wissen is distributed under [MIT License](LICENSE.txt)

