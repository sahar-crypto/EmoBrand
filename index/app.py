from flask import Flask, request, render_template, jsonify
from ntscraper import Nitter
import json
import pandas as pd 
import re
import os
import sys
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from deep_translator import GoogleTranslator


def translate_to_english(text, source_language='auto'):
    translator = GoogleTranslator(source=source_language, target='en')
    translation = translator.translate(text)
    return translation

def lowercase_text(text):
    if isinstance(text, str):
        return text.lower()
    else:
        # If the input is not a string (e.g., float or non-string), return it unchanged
        return text
    

def remove_com_links(text):
    # Define a regular expression pattern to match ".com" and everything before and after it until a space
    pattern = r'\S*\.com\S*'
    
    # Use re.sub to replace the matched pattern with an empty string
    cleaned_text = re.sub(pattern, '', text)
    
    return cleaned_text


def remove_html_tags(text):
    if isinstance(text, str):
        clean_text = re.sub(r'<.*?>', ' ', text)
        return clean_text
    else:
        return text
    

def remove_urls(text):
    if isinstance(text, str):
        # Remove any URL starting with "http://" or "https://"
        clean_text = re.sub(r'http\S+|www\S+', ' ', text)
        return clean_text
    else:
        return text
        
def remove_hashtags(text):
     if isinstance(text, str):
        return re.sub(r'#', ' ', text)
     else:
        # If the input is not a string (e.g., float or non-string), return it unchanged
        return text

    

# Remove mentions
def remove_mentions(text):
     if isinstance(text, str):
        return re.sub(r'@\w+', ' ', text)
     else:
        # If the input is not a string (e.g., float or non-string), return it unchanged
        return text

def convert_emojis(text):
    # Emoji mapping
    if isinstance(text, str):
        # Replace emojis with their text representations
        for emoji in emoji_mapping:
            text = text.replace(emoji, emoji_mapping[emoji])
        return text
    else:
        # If the input is not a string (e.g., float or non-string), return it unchanged
        return text

def convert_emoticons(text):
    # Emoticon mapping
     if isinstance(text, str):
        # Replace emojis with their text representations
        for emoticon in emoticon_mapping:
            text = text.replace(emoticon, emoticon_mapping[emoticon])
        return text
     else:
        # If the input is not a string (e.g., float or non-string), return it unchanged
        return text



emoticon_mapping = {
 '(:': 'Happy ',
 ':‑)': 'Happy ',
 ':-))': 'Very happy ',
 ':-)))': 'Very very Happy ',
 ':)': 'smiley ',
 ':))': 'Very smiley ',
 ':)))': 'Very very smiley ',
 ':-]': 'smiley ',
 ':]': 'Happy ',
 ':-3': 'smiley ',
 ':3': 'Happy ',
 ':->': 'smiley ',
 ':>': 'Happy ',
 '8-)': 'smiley ',
 ':o)': 'Happy ',
 ':-}': 'smiley ',
 ':}': 'Happy ',
 ':-)': 'smiley ',
 ':c)': 'Happy ',
 ':^)': 'smiley ',
 '=]': 'Happy ',
 '=)': 'smiley ',
 ':‑D': 'Laughing ',
 ':D': 'Laughing ',
 '8‑D': 'Laughing ',
 '8D': 'big grin ',
 'X‑D': 'big grin ',
 'XD': 'big grin ',
 '=D': 'laugh ',
 '=3': 'laugh ',
 'B^D': 'laugh ',
 ':-(': 'Frown ',
 ':‑(': 'Frown ',
 ':(': 'sad ',
 ':‑c': 'sad ',
 ':c': 'andry ',
 ':‑<': 'andry ',
 ':<': 'pouting ',
 ':‑[': 'pouting ',
 ':[': 'Frown ',
 ':-||': 'Frown ',
 '>:[': 'sad ',
 ':{': 'andry ',
 ':@': 'pouting ',
 '>:(': 'Frown ',
 ":'‑(": 'Crying ',
 ":'(": 'Crying ',
 ":'‑)": 'happiness ',
 ":')": 'happiness',
 "D‑':": 'Horror ',
 'D:<': 'Disgust ',
 'D:': 'Sadness ',
 'D8': 'dismay ',
 'D;': 'dismay ',
 'D=': 'dismay ',
 'DX': 'dismay ',
 ':‑O': 'Surprise ',
 ':O': 'Surprise ',
 ':‑o': 'Surprise ',
 ':o': 'Surprise ',
 ':-0': 'Shock ',
 '8‑0': 'Yawn ',
 '>:O': 'Yawn ',
 ':-*': 'Kiss ',
 ':*': 'Kiss ',
 ':X': 'Kiss ',
 ';‑)': 'Wink ',
 ';)': 'smirk ',
 '*-)': 'Wink ',
 '*)': 'smirk ',
 ';‑]': 'Wink ',
 ';]': 'smirk ',
 ';^)': 'Wink ',
 ':‑,': 'smirk ',
 ';D': 'Wink ',
 ':‑P': 'cheeky ',
 ':P': 'cheeky ',
 'X‑P': 'playful ',
 'XP': 'cheeky ',
 ':‑Þ': 'cheeky ',
 ':Þ': 'cheeky ',
 ':b': 'playful ',
 'd:': 'playful ',
 '=p': 'playful ',
 '>:P': 'playful ',
 ':‑/': 'Skeptical ',
 ':/': 'Skeptical ',
 ':-[.]': 'annoyed ',
 '>:[(\\)]': 'annoyed ',
 '>:/': 'undecided ',
 ':[(\\)]': 'undecided ',
 '=/': 'uneasy ',
 '=[(\\)]': 'uneasy ',
 ':L': 'hesitant ',
 '=L': 'hesitant ',
 ':S': 'Skeptical ',
 ':‑|': 'no expression ',
 ':|': 'indecision ',
 ':$': 'Embarrassed or blushing ',
 ':‑x': 'Sealed lips ',
 ':x': 'tongue-tied ',
 ':‑#': 'Sealed lips ',
 ':#': 'tongue-tied ',
 ':‑&': 'Sealed lips ',
 ':&': 'tongue-tied ',
 'O:‑)': 'Angel ',
 'O:)': 'innocent ',
 '0:‑3': 'Angel ',
 '0:3': 'innocent ',
 '0:‑)': 'Angel ',
 '0:)': 'innocent ',
 ':‑b': 'playful ',
 '0;^)': 'Angel ',
 '>:‑)': 'Evil ',
 '>:)': 'devilish ',
 '}:‑)': 'Evil ',
 '}:)': 'devilish ',
 '3:‑)': 'Evil ',
 '3:)': 'devilish ',
 '>;)': 'Evil ',
 '|;‑)': 'Cool ',
 '|‑O': 'Bored ',
 ':‑J': 'contempt ',
 '%‑)': 'Drunk ',
 '%)': 'confused ',
 ':-###..': 'sick ',
 ':###..': 'sick ',
 '<:‑|': 'Dump ',
 '(>_<)': 'Troubled ',
 '(>_<)>': 'Troubled ',
 "(';')": 'Baby ',
 '(^^>``': 'Nervous ',
 '(^_^;)': 'Embarrassed ',
 '(-_-;)': 'Troubled ',
 '(~_~;) (・.・;)': 'Shyp ',
 '(-_-)zzz': 'Sleeping ',
 '(^_-)': 'Wink ',
 '((+_+))': 'Confused ',
 '(+o+)': 'Confused ',
 '(o|o)': 'Ultraman ',
 '^_^': 'Joyful ',
 '(^_^)/': 'Joyful ',
 '(^O^)／': 'Joyful ',
 '(^o^)／': 'Joyful ',
 '(__)': 'respect ',
 '_(._.)_': 'apology ',
 '<(_ _)>': 'respect ',
 '<m(__)m>': 'apology ',
 'm(__)m': 'respect ',
 'm(_ _)m': 'apology ',
 "('_')": 'Sad ',
 '(/_;)': 'Crying ',
 '(T_T) (;_;)': 'Sad ',
 '(;_;': 'Crying ',
 '(;_:)': 'Sad ',
 '(;O;)': 'Crying ',
 '(:_;)': 'Sad ',
 '(ToT)': 'Crying ',
 ';_;': 'Sad ',
 ';-;': 'Crying ',
 ';n;': 'Sad ',
 ';;': 'Crying ',
 'Q.Q': 'Sad ',
 'T.T': 'Crying ',
 'QQ': 'Sad ',
 'Q_Q': 'Crying ',
 '(-.-)': 'Shame ',
 '(-_-)': 'Shame ',
 '(一一)': 'Shame ',
 '(；一_一)': 'Shame ',
 '(=_=)': 'Tired ',
 '(=^·^=)': 'cat ',
 '(=^··^=)': 'cat ',
 '=_^= ': 'cat ',
 '(..)': 'sadness ',
 '(._.)': 'boredom ',
 '^m^': 'Giggling ',
 '(・・?': 'Confusion ',
 '(?_?)': 'Confusion ',
 '>^_^<': 'Laugh ',
 '<^!^>': 'Laugh ',
 '^/^': 'Laugh ',
 '（*^_^*）': 'Laugh ',
 '(^<^) (^.^)': 'Laugh ',
 '(^^)': 'Laugh ',
 '(^.^)': 'Laugh ',
 '(^_^.)': 'Laugh ',
 '(^_^)': 'Laugh ',
 '(^J^)': 'Laugh ',
 '(*^.^*)': 'Laugh ',
 '(^—^）': 'Laugh ',
 '(#^.^#)': 'Laugh ',
 '（^—^）': 'Waving ',
 '(;_;)/~~~': 'Waving ',
 '(^.^)/~~~': 'Waving ',
 '(-_-)/~~~ ($··)/~~~': 'Waving ',
 '(T_T)/~~~': 'Waving ',
 '(ToT)/~~~': 'Waving ',
 '(*^0^*)': 'Excited ',
 '(*_*)': 'Amazed ',
 '(*_*;': 'Amazed ',
 '(+_+) (@_@)': 'Amazed ',
 '(*^^)v': 'Laughing ',
 '(^_^)v': 'Cheerful ',
 '((d[-_-]b))': 'Headphones ',
 '(-"-)': 'Worried ',
 '(ーー;)': 'Worried ',
 '(^0_0^)': 'win ',
 '(＾ｖ＾)': 'Happy ',
 '(＾ｕ＾)': 'Happy ',
 '(^)o(^)': 'Happy ',
 '(^O^)': 'Happy ',
 '(^o^)': 'Happy ',
 ')^o^(': 'Happy ',
 ':O o_O': 'Surprised ',
 'o_0': 'Surprised ',
 'o.O': 'Surpised ',
 '(o.o)': 'Surprised ',
 'oO': 'Surprised ',
 '(*￣m￣)': 'Dissatisfied ',
 '(‘A`)': 'Deflated ',
  '*-*':'In Love ',
  '^^':'happy ',
  'c:':'bummed',
  '( ´ ▽ ` )ﾉ':"happy",
  '(:':'smile',
   '>.<':'annoyed',
   '-_-':'neutral'
     
}

# emoji mapping
emoji_mapping = {
    '🙂':'Smiley ',
    '😊':'happy ',
    '😀':'Smiley ',
    '😁':'happy ',
    '😃':'Laughing ',
    '😄':'big grin ',
    '😆':'Laughing ',
    '😂':'Laughing ',
    '🤒':'sick ',
    '😛':'playful ',
    '☹️':'Frown ',
    '🙁':'sad ',
    '😔':'sad ',
    '😞':'pouting ',
    '😟':'Frown ',
    '😣':'annoyed ',
    '😖':'hesitant ',
    '😢':'Crying ',
    '😭':'Crying ',
    '🥺':'crying ',
    '😠':'Angry ',
    '😡':'Angry ',
    '😨':'Horror ',
    '😧':'Horror ',
    '😱':'Shocked ',
    '😫':'sadness ',
    '😩':'dismay ',
    '😦':'sadness ',
    '😮':'Surprise ',
    '😯':'Surprise ',
    '😲':'shock ',
    '😗':'Kiss ',
    '😙':'Kiss ',
    '😚':'Kiss',
    '😘':'Kiss ',
    '😍':'love ',
    '😉':'Wink ',
    '😜':'smirk ',
    '😝':'cheeky ',
    '😜':'playful ',
    '🤑':'money ',
    '😐':'no expression ',
    '😑':'indecision ',
    '😳':'Embarrassed ',
    '🤐':'Sealed lips ',
    '😶':'tongue tied ',
    '😇':'Angel ',
    '👼':'innocent ',
    '😈':'Evil ',
    '😎':'Cool ',
    '😪':'bored ',
    '😏':'disdain ',
    '😒':'disdain ',
    '😕':'confused ',
    '😵‍':'Drunk ',
    '🤕':'confused ',
    '🤒':'sick ',
    '😷':'sick ',
    '🤢':'disgust ',
    '🤨':'Scepticism ',
    '😬':'Grimacing ',
    '☠️':'dangerous ',
    '💀':'grave ',
    '🌹':'love ',
    '❤️':'love ',
    '💔':'sad ',
    '🍻':'Cheer ',
    '👶':'Baby ',
    '😅':'troubled ',
    '😓':'disappointed ',
    '😴':'Sleeping ',
    '💤':'Sleeping ',
    '🙄':'Confused ',
    '🙌':'Joyful ',
    '🙇':'apology ',
    '💃':'Excited ',
    '🤷':'shrug ',
}

emo_mapper = {
    'angry': 'negative',
    'sadness': 'negative',
    'disgust': 'negative',
    'fear': 'negative',
    'joy': 'positive',
    'surprise': 'positive',
    'neutral': 'positive'
}


def get_posts(brand):
    
    query = str(brand)
    
    scraper = Nitter()
    
    tweets = scraper.get_tweets(query, mode='hashtag', number=100)
    
    text = []
    
    date = []
    
    for tweet in tweets['tweets']:
        text.append(tweet['text'])
        date.append(tweet['date'])
    
    df = pd.DataFrame({'text': text})
    
    #df['date'].fillna(df['date'].mode().iloc[0], inplace=True)
    
    df['text'] = df['text'].apply(translate_to_english)
    df['text'] = df['text'].apply(remove_urls)
    df['text'] = df['text'].apply(remove_com_links)
    df['text'] = df['text'].apply(convert_emojis)
    df['text'] = df['text'].apply(convert_emoticons)
    df['text'] = df['text'].apply(remove_mentions)
    df['text'] = df['text'].apply(remove_hashtags)
    df['text'] = df['text'].apply(remove_html_tags)
    df['text'] = df['text'].apply(lowercase_text)
    
    
    return df

def get_predictions(data):
    
    model_path = "../model/"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    inputs = tokenizer(data['text'].tolist(), padding=True, truncation=True, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

    predicted_classes = torch.argmax(predictions, dim=-1).tolist()
    
    labels = {0: 'anger',
              1: 'disgust',
              2: 'fear',
              3: 'joy',
              4: 'neutral',
              5: 'sadness',
              6: 'surprise'}
    
    data['predicted_label'] = predicted_classes
    data['predicted_label'] = data['predicted_label'].map(labels)
    
    data['sentiment'] = data['predicted_label'].map(emo_mapper)
    
    emo_percentage = data.predicted_label.value_counts()
    
    sentiment = data.sentiment.value_counts().index[0]
    
    recom = {}
    if (sentiment=="positive" ):
        recom["Customer Service"] = "Provide exceptional and personalized service. Remember the customer's name, preferences, and previous interactions. Offer proactive assistance and go the extra mile to exceed their expectations."
        recom["Product Quality"] = "Consistently maintain a high-quality product. Use customer feedback to improve and innovate. Highlight product features that make customers happy in marketing materials."
        recom["Price"] = "Offer competitive prices and discounts for loyal customers. Create loyalty programs or bundles to reward repeat business"
        recom["Marketing"] = "Leverage happy customer testimonials and case studies in marketing materials. Use social proof to highlight the positive experiences of others."
    else:
        recom["Customer Service"] = "Address their concerns promptly and empathetically. Listen actively, apologize for any inconvenience, and offer solutions that solve their problem. Ensure follow-up to make sure the issue is resolved to their satisfaction."
        recom["Product Quality"] = "Quickly address quality issues by implementing effective quality control measures. Replace or refund defective products without hesitation, and communicate openly about improvements made to avoid similar issues in the future."
        recom["Price"] = "Provide transparent pricing with no hidden costs. If a customer is unhappy with the price, explore flexible payment options or consider offering a price match guarantee."
        recom["Marketing"] = "Use negative feedback constructively. Address customer concerns in your marketing strategy by emphasizing improvements made based on their feedback. Show your commitment to making the customer's experience better."
    
    
    emo_dict = {
        idx: val for (idx, val) in zip(emo_percentage.index, emo_percentage)
    }
    
    sample_tweet = data.text.sample(1).iloc[0]
    
    response = {
        'emo_dict': emo_dict,
        'recom': recom,
        'sample_tweet': sample_tweet
    }
    
    return response

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/run_script', methods=['POST'])
def run_script():
    data = request.get_json()
    brand = data.get('brand') 
    data = get_posts(brand)
    response = get_predictions(data)
    return jsonify({'body': response})

if __name__ == '__main__':
    app.run(debug=True)
