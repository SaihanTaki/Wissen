<h1 style="text-align: center">Wissen</h1> 

#### *In  German, `Wissen` means `Knowledge`*

## 💡 About
Wissen is a  LLM-powered chatbot that can handle conversations across multiple documents.
This app is built with
- Langchain
- Google Palm
- Streamlit

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

