from flask import Flask,render_template,request
from transliteration import decode_sequence,input_token_index
import re
import numpy as np


app=Flask(__name__)
max_encoder_seq_length=25
num_encoder_tokens=len(input_token_index)

@app.route('/')
def index():
   return render_template('index.html')


@app.route('/transliteration',methods=['POST'])
def transliteration():
    if request.method == 'POST':
      text_input = request.form['text'].strip()
      pattern = r'[^a-zA-Z\s]'
      text_input = re.sub(pattern, '', text_input)
      text_list = text_input.split()
      print(text_list)
      encoder_input_data = np.zeros(
      (len(text_list), max_encoder_seq_length, num_encoder_tokens), dtype="float32")

      print(encoder_input_data.shape)
      

      for i, input_text in enumerate(text_list):
         for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.0
         encoder_input_data[i, t + 1 :, input_token_index[" "]] = 1.0

      decoded_sentences = []
      for input_data in encoder_input_data:
         decoded_sentence = decode_sequence(input_data[np.newaxis, :, :])
         decoded_sentences.append(decoded_sentence.strip())
      decoded_sentences

      return render_template('index.html',result=" ".join(decoded_sentences))



    


if __name__=="__main__":
    app.run(port=8002)