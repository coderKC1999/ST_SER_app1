import streamlit as st

import torch

import librosa

import numpy as np

import soundfile as sf

from PIL import Image

from torchvision import transforms

from audio_recorder_streamlit import audio_recorder

from model import HybridResNetViT

from utils import *

#from emotion_classes import emotion_classes


emotion_classes = [
"Anger","calm","Disgust","Fear",
"Happyness","neutral","Sadness","Surprised"
]

st.set_page_config(

    page_title="SER",

    layout="centered"

)

st.title(

    "Speech Emotion Recognition"

)


@st.cache_resource
def load_model():

    model = HybridResNetViT(

        num_classes=8

    )

    model.load_state_dict(

        torch.load(

            "RAVDESS_ST_ResViT_02_8576.pth",

            map_location="cpu"

        )

    )

    model.eval()

    return model


model = load_model()


transform = transforms.Compose([

    transforms.Resize(

        (224,224)

    ),

    transforms.ToTensor(),

    transforms.Normalize(

        [0.485,0.456,0.406],

        [0.229,0.224,0.225]

    )

])


option = st.radio(

    "Choose input",

    [

        "Record Audio",

        "Upload Audio",
        
        "Upload Spectrogram Image"

    ]

)

audio_path = None


if option == "Record Audio":

    audio_bytes = audio_recorder()

    if audio_bytes:

        audio_path = "record.wav"

        with open(

            audio_path,

            "wb"

        ) as f:

            f.write(

                audio_bytes

            )
        st.audio(audio_bytes)


elif option == "Upload Audio":

    uploaded = st.file_uploader(

        "Upload wav",

        type=["wav"]

    )

    if uploaded:

        audio_path = "upload.wav"

        with open(

            audio_path,

            "wb"

        ) as f:

            f.write(

                uploaded.getbuffer()

            )
        st.audio(uploaded)
		

if audio_path:

    y,sr = librosa.load(

        audio_path,

        sr=8000

    )

    # keep last 3 seconds

    samples = 3*sr

    if len(y) > samples:

        y = y[-samples:]
    
    st.write("Audio samples:", len(y))
    st.write("Sampling rate:", sr)
    S,f,t = stockwell_spectrogram(

        y,

        sr

    )
    st.write("S shape:", S.shape)
    st.write("f shape:", f.shape)
    st.write("t shape:", t.shape)

    image_path = "temp.png"

    save_spectrogram(

        S,

        f,

        t,

        image_path

    )


    image = Image.open(

        image_path

    ).convert(

        "RGB"

    )

    st.image(

        image,

        caption="Stockwell Spectrogram"

    )


    x = transform(

        image

    )

    x = x.unsqueeze(

        0

    )


    with torch.no_grad():

        output = model(

            x

        )

        probs = torch.softmax(

            output,

            dim=1

        )

        conf,pred = torch.max(

            probs,

            dim=1

        )


    emotion = emotion_classes[

        pred.item()

    ]


    st.success(

        f"Emotion : {emotion}"

    )


    st.write(

        f"Confidence : {conf.item()*100:.2f}%"

    )


    st.bar_chart(

        probs.numpy().flatten()

    )


elif option == "Upload Spectrogram Image":

    image_file = st.file_uploader(

        "Upload image",

        type=["png", "jpg", "jpeg"]

    )

    if image_file:

        image = Image.open(

            image_file

        ).convert("RGB")

        st.image(

            image,

            caption="Uploaded Spectrogram"

        )

        x = transform(image)

        x = x.unsqueeze(0)

        with torch.no_grad():

            outputs = model(x)

            probs = torch.softmax(

                outputs,

                dim=1

            )

            conf, pred = torch.max(

                probs,

                dim=1

            )

        emotion = emotion_classes[

            pred.item()

        ]

        st.success(

            f"Emotion: {emotion}"

        )

        st.write(

            f"Confidence: {conf.item()*100:.2f}%"

        )

        st.bar_chart(

            probs.numpy().flatten()

        )
        
