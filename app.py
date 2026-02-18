{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "21da40ff-43e3-4cac-bae1-f9a18ab70cfb",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "5950db74-5a04-4482-870c-cdb35061874e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: gradio in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (5.49.0)\n",
      "Requirement already satisfied: aiofiles<25.0,>=22.0 in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from gradio) (24.1.0)\n",
      "Requirement already satisfied: anyio<5.0,>=3.0 in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from gradio) (4.9.0)\n",
      "Requirement already satisfied: brotli>=1.1.0 in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from gradio) (1.1.0)\n",
      "Requirement already satisfied: fastapi<1.0,>=0.115.2 in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from gradio) (0.118.1)\n",
      "Requirement already satisfied: ffmpy in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from gradio) (0.6.2)\n",
      "Requirement already satisfied: gradio-client==1.13.3 in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from gradio) (1.13.3)\n",
      "Requirement already satisfied: groovy~=0.1 in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from gradio) (0.1.2)\n",
      "Requirement already satisfied: httpx<1.0,>=0.24.1 in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from gradio) (0.28.1)\n",
      "Requirement already satisfied: huggingface-hub<2.0,>=0.33.5 in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from gradio) (0.35.3)\n",
      "Requirement already satisfied: jinja2<4.0 in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from gradio) (3.1.6)\n",
      "Requirement already satisfied: markupsafe<4.0,>=2.0 in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from gradio) (3.0.2)\n",
      "Requirement already satisfied: numpy<3.0,>=1.0 in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from gradio) (2.2.0)\n",
      "Requirement already satisfied: orjson~=3.0 in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from gradio) (3.11.0)\n",
      "Requirement already satisfied: packaging in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from gradio) (25.0)\n",
      "Requirement already satisfied: pandas<3.0,>=1.0 in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from gradio) (2.3.3)\n",
      "Requirement already satisfied: pillow<12.0,>=8.0 in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from gradio) (11.3.0)\n",
      "Requirement already satisfied: pydantic<2.12,>=2.0 in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from gradio) (2.11.7)\n",
      "Requirement already satisfied: pydub in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from gradio) (0.25.1)\n",
      "Requirement already satisfied: python-multipart>=0.0.18 in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from gradio) (0.0.20)\n",
      "Requirement already satisfied: pyyaml<7.0,>=5.0 in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from gradio) (6.0.2)\n",
      "Requirement already satisfied: ruff>=0.9.3 in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from gradio) (0.14.0)\n",
      "Requirement already satisfied: safehttpx<0.2.0,>=0.1.6 in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from gradio) (0.1.6)\n",
      "Requirement already satisfied: semantic-version~=2.0 in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from gradio) (2.10.0)\n",
      "Requirement already satisfied: starlette<1.0,>=0.40.0 in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from gradio) (0.48.0)\n",
      "Requirement already satisfied: tomlkit<0.14.0,>=0.12.0 in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from gradio) (0.13.3)\n",
      "Requirement already satisfied: typer<1.0,>=0.12 in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from gradio) (0.17.4)\n",
      "Requirement already satisfied: typing-extensions~=4.0 in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from gradio) (4.14.1)\n",
      "Requirement already satisfied: uvicorn>=0.14.0 in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from gradio) (0.37.0)\n",
      "Requirement already satisfied: fsspec in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from gradio-client==1.13.3->gradio) (2025.3.0)\n",
      "Requirement already satisfied: websockets<16.0,>=13.0 in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from gradio-client==1.13.3->gradio) (15.0.1)\n",
      "Requirement already satisfied: idna>=2.8 in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from anyio<5.0,>=3.0->gradio) (3.10)\n",
      "Requirement already satisfied: sniffio>=1.1 in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from anyio<5.0,>=3.0->gradio) (1.3.1)\n",
      "Requirement already satisfied: certifi in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from httpx<1.0,>=0.24.1->gradio) (2025.6.15)\n",
      "Requirement already satisfied: httpcore==1.* in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from httpx<1.0,>=0.24.1->gradio) (1.0.9)\n",
      "Requirement already satisfied: h11>=0.16 in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from httpcore==1.*->httpx<1.0,>=0.24.1->gradio) (0.16.0)\n",
      "Requirement already satisfied: filelock in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from huggingface-hub<2.0,>=0.33.5->gradio) (3.18.0)\n",
      "Requirement already satisfied: requests in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from huggingface-hub<2.0,>=0.33.5->gradio) (2.32.4)\n",
      "Requirement already satisfied: tqdm>=4.42.1 in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from huggingface-hub<2.0,>=0.33.5->gradio) (4.67.1)\n",
      "Requirement already satisfied: python-dateutil>=2.8.2 in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from pandas<3.0,>=1.0->gradio) (2.9.0.post0)\n",
      "Requirement already satisfied: pytz>=2020.1 in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from pandas<3.0,>=1.0->gradio) (2025.2)\n",
      "Requirement already satisfied: tzdata>=2022.7 in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from pandas<3.0,>=1.0->gradio) (2025.2)\n",
      "Requirement already satisfied: annotated-types>=0.6.0 in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from pydantic<2.12,>=2.0->gradio) (0.7.0)\n",
      "Requirement already satisfied: pydantic-core==2.33.2 in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from pydantic<2.12,>=2.0->gradio) (2.33.2)\n",
      "Requirement already satisfied: typing-inspection>=0.4.0 in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from pydantic<2.12,>=2.0->gradio) (0.4.1)\n",
      "Requirement already satisfied: click>=8.0.0 in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from typer<1.0,>=0.12->gradio) (8.2.1)\n",
      "Requirement already satisfied: shellingham>=1.3.0 in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from typer<1.0,>=0.12->gradio) (1.5.4)\n",
      "Requirement already satisfied: rich>=10.11.0 in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from typer<1.0,>=0.12->gradio) (14.1.0)\n",
      "Requirement already satisfied: colorama in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from click>=8.0.0->typer<1.0,>=0.12->gradio) (0.4.6)\n",
      "Requirement already satisfied: six>=1.5 in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from python-dateutil>=2.8.2->pandas<3.0,>=1.0->gradio) (1.17.0)\n",
      "Requirement already satisfied: markdown-it-py>=2.2.0 in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from rich>=10.11.0->typer<1.0,>=0.12->gradio) (3.0.0)\n",
      "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from rich>=10.11.0->typer<1.0,>=0.12->gradio) (2.19.2)\n",
      "Requirement already satisfied: mdurl~=0.1 in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from markdown-it-py>=2.2.0->rich>=10.11.0->typer<1.0,>=0.12->gradio) (0.1.2)\n",
      "Requirement already satisfied: charset_normalizer<4,>=2 in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from requests->huggingface-hub<2.0,>=0.33.5->gradio) (3.4.2)\n",
      "Requirement already satisfied: urllib3<3,>=1.21.1 in c:\\users\\kiit0001\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from requests->huggingface-hub<2.0,>=0.33.5->gradio) (2.5.0)\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n",
      "[notice] A new release of pip is available: 25.3 -> 26.0.1\n",
      "[notice] To update, run: C:\\Users\\KIIT0001\\AppData\\Local\\Programs\\Python\\Python311\\python.exe -m pip install --upgrade pip\n"
     ]
    }
   ],
   "source": [
    "%pip install gradio"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "415c56dd-cbd8-4cd6-93d7-8535d0222eb9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "* Running on local URL:  http://127.0.0.1:7861\n",
      "* To create a public link, set `share=True` in `launch()`.\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div><iframe src=\"http://127.0.0.1:7861/\" width=\"100%\" height=\"500\" allow=\"autoplay; camera; microphone; clipboard-read; clipboard-write;\" frameborder=\"0\" allowfullscreen></iframe></div>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": []
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import gradio as gr\n",
    "import joblib\n",
    "import numpy as np\n",
    "\n",
    "model = joblib.load(\"model.pkl\")\n",
    "vectorizer = joblib.load(\"vectorizer.pkl\")\n",
    "label_encoder = joblib.load(\"label_encoder.pkl\")\n",
    "\n",
    "def predict_user(text):\n",
    "    if text.strip() == \"\":\n",
    "        return \"Enter a message\", None\n",
    "    \n",
    "    vec = vectorizer.transform([text])\n",
    "    pred = model.predict(vec)[0]\n",
    "    probs = model.predict_proba(vec)[0]\n",
    "    \n",
    "    user = label_encoder.inverse_transform([pred])[0]\n",
    "    \n",
    "    confidence = float(np.max(probs))\n",
    "    \n",
    "    return user, confidence\n",
    "\n",
    "with gr.Blocks() as demo:\n",
    "    \n",
    "    gr.Markdown(\"# Conversational Intelligence System\")\n",
    "    gr.Markdown(\"Predict the likely sender of a message\")\n",
    "    \n",
    "    text = gr.Textbox(label=\"Enter Message\")\n",
    "    \n",
    "    btn = gr.Button(\"Predict\")\n",
    "    \n",
    "    user_out = gr.Textbox(label=\"Predicted User\")\n",
    "    conf_out = gr.Number(label=\"Confidence Score\")\n",
    "    \n",
    "    btn.click(predict_user, inputs=text, outputs=[user_out, conf_out])\n",
    "\n",
    "demo.launch()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3e923e55-872f-46e3-8d28-d167666bad6d",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
