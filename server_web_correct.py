from flask import Flask, render_template, request
# Tạo ứng dụng Flask
from collections import Counter
from keras.models import load_model
import re
import numpy as np
from nltk import ngrams
import nltk
from unidecode import unidecode
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Activation, Bidirectional
from tensorflow.keras.optimizers import Adam
import time


app = Flask(__name__)

# import matplotlib.pyplot as plt

vowel = list('aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴ')
full_letters = vowel + list('bcdfghjklmnpqrstvwxzBCDFGHJKLMNPQRSTVWXZđĐ')


"""
Mỗi ký tự đặc biệt (có dấu) được ánh xạ tới một chuỗi ký tự tương ứng. Ví dụ, 'á' được ánh xạ thành 'as'
"""
typo = {
    'á': 'as', 'Á': 'As', 'à': 'af', 'À': 'Af', 'ả': 'ar', 'Ả': 'Ar', 'ã': 'ax', 'Ã': 'Ax', 'ạ': 'aj', 'Ạ': 'Aj',
    'ắ': 'aws', 'Ắ': 'Aws', 'ằ': 'awf', 'Ằ': 'Awf', 'ẳ': 'awr', 'Ẳ': 'Awr', 'ẵ': 'awx', 'Ẵ': 'Awx', 'ặ': 'awj', 'Ặ': 'Awj', 'ă': 'aw', 'Ă':'Aw',
    'ấ': 'aas', 'Ấ': 'Aas', 'ầ': 'aaf', 'Ầ': 'Aaf', 'ẩ': 'aar', 'Ẩ': 'Aar', 'ẫ': 'aax', 'Ẫ': 'Aax', 'ậ': 'aaj', 'Ậ': 'Aaj', 'â': 'aa', 'Â': 'Aa',
    'é': 'es', 'É': 'Es', 'è':'ef', 'È': 'Ef', 'ẻ': 'er', 'Ẻ': 'Er', 'ẽ': 'ex', 'Ẽ': 'Ex', 'ẹ': 'ej', 'Ẹ': 'Ej',
    'ế': 'ees', 'Ế': 'Ees', 'ề': 'eef', 'Ề': 'Eef', 'ể': 'eer', 'Ể': 'Eer', 'ễ': 'eex', 'Ễ': 'Eex', 'ệ': 'eej', 'Ệ': 'Eej', 'ê': 'ee', 'Ê': 'Ee',
    'í': 'is', 'Í': 'Is', 'ì': 'if', 'Ì':'If', 'ỉ': 'ir', 'Ỉ': 'Ir', 'ĩ': 'ix', 'Ĩ': 'Ix', 'ị': 'ij', 'Ị': 'Ij',
    'ó': 'os', 'Ó': 'Os', 'ò': 'of', 'Ò': 'Of', 'ỏ': 'or', 'Ỏ': 'Or', 'õ': 'ox', 'Õ': 'ox', 'ọ': 'oj', 'Ọ': 'Oj',
    'ố': 'oos', 'Ố': 'Oos', 'ồ': 'oof', 'Ồ': 'Oof', 'ổ': 'oor', 'Ổ': 'Oor', 'ỗ': 'oox', 'Ỗ': 'Oox', 'ộ': 'ooj', 'Ộ': 'Ooj', 'ô': 'oo', 'Ô': 'Oo',
    'ớ': 'ows', 'Ớ': 'Ows', 'ờ': 'owf', 'Ờ': 'OwF', 'ở': 'owr', 'Ở': 'Owr', 'ỡ': 'owx', 'Ỡ': 'Owx', 'ợ': 'owj', 'Ợ': 'Owj', 'ơ': 'ow', 'Ơ': 'Ow',
    'ú': 'us', 'Ú': 'Us', 'ù': 'uf', 'Ù': 'Uf', 'ủ': 'ur', 'Ủ': 'Ur', 'ũ': 'ux', 'Ũ': 'Ux', 'ụ': 'uj', 'Ụ': 'Uj',
    'ứ': 'aws', 'Ứ': 'Uws', 'ừ': 'uwf', 'Ừ': 'Uwf', 'ử': 'uwr', 'Ử': 'Uwr', 'ữ': 'uwx', 'Ữ': 'Uwx', 'ự': 'uwj', 'Ự': 'Uwj', 'ư': 'uw', 'Ư': 'Uw',
    'ý': 'ys', 'Ý': 'Ys', 'ỳ': 'yf', 'Ỳ': 'Yf', 'ỷ': 'yr', 'Ỷ': 'Yr', 'ỹ': 'yx', 'Ỹ': 'Yx', 'ỵ': 'yj', 'Ỵ': 'Yj',
    'đ': 'dd', 'Đ': 'Dd',
}

region = {
    'ả': 'ã', 'ã': 'ả', 'ẻ': 'ẽ', 'ẽ': 'ẻ', 'ỉ': 'ĩ', 'ĩ': 'ỉ', 'ỏ': 'õ', 'õ': 'ỏ', 'ở': 'ỡ', 'ỡ': 'ở', 'ổ': 'ỗ', 'ỗ': 'ổ', 'ủ': 'ũ', 'ũ': 'ủ', 'ử': 'Ữ', 'ữ': 'ử', 
    'ỷ': 'ỹ', 'ỹ': 'ỷ', 
    'Ả': 'Ã', 'Ã': 'Ả', 'Ẻ': 'Ẽ', 'Ẽ': 'Ẻ', 'Ỉ': 'Ĩ', 'Ĩ': 'Ỉ', 'Ỏ': 'Õ', 'Õ': 'Ỏ', 'Ở': 'Ỡ', 'Ỡ': 'Ở', 'Ổ': 'Ỗ', 'Ỗ': 'Ổ', 'Ủ': 'Ũ', 'Ũ': 'Ủ', 'Ử': 'Ữ', 'Ữ': 'Ử',
    'Ỷ': 'Ỹ', 'Ỹ': 'Ỷ', 
}

region2 = {
    'ch': 'tr', 'tr': 'ch', 
    'Ch': 'Tr', 'Tr': 'Ch',
    'd': 'gi', 'gi': 'd', 
    'D': 'Gi', 'Gi': 'D',
    'l': 'n', 'n': 'l', 
    'L': 'N', 'N': 'L',
    'x': 's', 's': 'x', 
    'X': 'S', 'S': 'X'
}

"""
Mỗi từ thường được viết tắt thành một chuỗi ký tự ngắn hơn. Ví dụ, 'anh' được viết tắt thành 'a'
"""
acronym = {
    'anh': 'a', 'biết': 'bít', 'chồng': 'ck', 'được': 'dc', 'em': 'e', 'gì': 'j', 'giờ': 'h',
    'Anh': 'A', 'Biết': 'Bít', 'Chồng': 'Ck', 'Được': 'Dc', 'Em': 'E', 'Gì': 'J', 'Giờ': 'H',
    'không': 'ko', 'muốn': 'mún', 'ông': 'ôg', 'phải': 'fai', 'tôi': 't', 'vợ': 'vk', 'yêu': 'iu',
    'Không': 'Ko', 'Muốn': 'Mún', 'Ông': 'Ôg', 'Phải': 'Fai', 'Tôi': 'T', 'Vợ': 'Vk', 'Yêu': 'Iu',
}

def _teen_code(sentence):
    random = np.random.uniform(0,1,1)[0]
    new_sentence = str(sentence)

    if random > 0.5:
        for word in acronym.keys():
            # Tìm và thay thế từ hoặc cụm từ trong câu, không dùng biên giới từ (\b) cho các cụm từ có dấu cách
            if word in new_sentence:
                random2 = np.random.uniform(0,1,1)[0]
                if random2 < 0.5:
                    new_sentence = new_sentence.replace(word, acronym[word])
        return new_sentence
    else:
        return sentence


def _add_noise(sentence):
    sentence = _teen_code(sentence)
    noisy_sentence = ''
    i = 0

    while i < len(sentence):
        if sentence[i] not in full_letters:
            noisy_sentence += sentence[i]
        else:
            random = np.random.uniform(0,1,1)[0]
            if random <= 0.94:
                noisy_sentence += sentence[i]
            elif random <= 0.985:
                if sentence[i] in typo:
                    if sentence[i] in region:
                        random2 = np.random.uniform(0,1,1)[0]
                        if random2 <= 0.4:
                            noisy_sentence += ''.join(typo.get(sentence[i], [sentence[i]]))
                        elif random2 <= 0.8:
                            noisy_sentence += ''.join(region.get(sentence[i], [sentence[i]]))
                        elif random2 <= 0.95:
                            noisy_sentence += unidecode(sentence[i])
                        else:
                            noisy_sentence += sentence[i]
                    else:
                        noisy_sentence += ''.join(typo.get(sentence[i], [sentence[i]]))
                else:
                    random3 = np.random.uniform(0,1,1)[0]
                    if random3 <= 0.6:
                        noisy_sentence += ''.join(typo.get(sentence[i], [sentence[i]]))
                    elif random3 < 0.9:
                        noisy_sentence += unidecode(sentence[i])
                    else:
                        noisy_sentence += sentence[i]
            elif i == 0 or sentence[i-1] not in full_letters:
                random4 = np.random.uniform(0,1,1)[0]
                if random4 <= 0.9:
                    if i < len(sentence) - 1 and sentence[i] in region2.keys() and sentence[i+1] in vowel:
                        noisy_sentence += region2[sentence[i]]
                    elif i < len(sentence) - 2 and sentence[i:i+2] in region2.keys() and sentence[i+2] in vowel:
                        noisy_sentence += region2[sentence[i:i+2]]
                        i += 1
                    else:
                        noisy_sentence += sentence[i]
                else:
                    noisy_sentence += sentence[i]
            else:
                new_random = np.random.uniform(0, 1)
                if new_random <= 0.33 and i < len(sentence) - 1:
                    noisy_sentence += sentence[i+1]
                    noisy_sentence += sentence[i]
                    i += 1
                else:
                    noisy_sentence += sentence[i]
        i += 1
    return noisy_sentence

"""
Định nghĩa bộ ký tự (alphabet) mà mô hình sẽ sử dụng để mã hóa và giải mã văn bản.
Chi tiết:
'\x00': Thường được sử dụng như một ký tự đặc biệt (padding).
' ' (space): Để phân tách các từ.
Các chữ số từ 0 đến 9.
Các chữ cái tiếng Việt bao gồm cả nguyên âm và phụ âm.
"""


# alphabet = ['\x00', ' ', '/', '-'] + list('0123456789') + full_letters 

alphabet = ['\x00', ' '] + list('0123456789') + full_letters 
# alphabet = [' ', '/', '-', '|'] + list('0123456789') + full_letters


def _encoder_data(text):
  x = np.zeros((MAXLEN, len(alphabet)))
  for i, c in enumerate(text[:MAXLEN]):
    x[i, alphabet.index(c)] = 1
  if i <  MAXLEN - 1:
    for j in range(i+1, MAXLEN):
      x[j, 0] = 1
  return x

def _decoder_data(x):
  x = x.argmax(axis=-1)
  return ''.join(alphabet[i] for i in x)

model_path = 'E:\\CTU\\luan_van\\seqtoseq\\model_31_10_2024.h5'

# model_path = './model_31_10_2024.h5'

model = load_model(model_path)

NGRAM = 5
MAXLEN = 39


def _nltk_ngrams(sentence, n, maxlen):
    list_ngrams = []
    list_words = sentence.split()
    num_words = len(list_words)

    if num_words >= n:
        for ngram in nltk.ngrams(list_words, n):
            if len(' '.join(ngram)) <= maxlen:
                list_ngrams.append(ngram)
    else:
        list_ngrams.append(tuple(list_words))
    return list_ngrams



def _guess(ngram):
    # print("Input ngram:", ngram)
    text = " ".join(ngram)  
    text = re.sub(r"[^\w\s/-]", '', text)  # Giữ lại các ký tự chữ, số, khoảng trắng, '/' và '-'
    # print("Processed text:", text)
    preds = model.predict(np.array([_encoder_data(text)]))
    # print("Predictions from model:", preds)
    # Giải mã kết quả dự đoán
    result = _decoder_data(preds[0]).strip('\x00') 
    # In ra kết quả sau giải mã
    # print("Decoded result:", result)
    
    return result





def _add_punctuation(text, corrected_text):
    list_punctuation = {}
    for (i, word) in enumerate(text.split()):
        if word[0] not in alphabet or word[-1] not in alphabet:
            # Dấu ở đầu chữ như " và '
            start_punc = ''
            for c in word:
                if c in alphabet:
                    break
                start_punc += c

            # Dấu ở sau chữ như .?!,;
            end_punc = ''
            for c in word[::-1]:
                if c in alphabet:
                    break
                end_punc += c
            end_punc = end_punc[::-1]

            # Lưu vị trí từ và dấu câu trong từ đó
            list_punctuation[i] = [start_punc, end_punc]

    # Thêm dấu câu vào vị trí các từ đã đánh dấu
    result = ''
    for (i, word) in enumerate(corrected_text.split()):
        if i in list_punctuation:
            result += (list_punctuation[i][0] + word + list_punctuation[i][1]) + ' '
        else:
            result += word + ' '

    return result.strip()



def _correct(text):
    # Xóa các ký tự đặc biệt
    new_text = re.sub(r"[^" + ''.join(alphabet) + ']', '', text)
    # print("->>>>>>>>>>>>"+new_text)
    ngrams = list(_nltk_ngrams(new_text, NGRAM, MAXLEN))
    
    guessed_ngrams = list(_guess(ngram) for ngram in ngrams)
    candidates = [Counter() for _ in range(len(guessed_ngrams) + NGRAM - 1)]

    for nid, ngram in enumerate(guessed_ngrams):
        for wid, word in enumerate(re.split(r'\s', ngram)):
            index = nid + wid
            # print(f'nid: {nid}, wid: {wid}, index: {index}, candidates length: {len(candidates)}')
            if index < len(candidates):
                candidates[index].update([word])
            else:
                # Safely append to candidates if the index exceeds the current list length
                candidates.append(Counter([word]))
                # print(f"Index {index} is out of range, adding new Counter!")

    corrected_text = ' '.join(c.most_common(1)[0][0] for c in candidates if c)
    return _add_punctuation(text, corrected_text)

@app.route("/", methods=["GET", "POST"])
def index():
    text = ""  # Khởi tạo giá trị mặc định
    result = None
    processing_time = None
    error_message = None  # Biến chứa thông báo lỗi nếu có

    if request.method == "POST":
        text = request.form.get("text", "")  # Lấy dữ liệu từ form
        
        try:
            start_time = time.time()  # Bắt đầu đo thời gian
            result = _correct(text)  # Hàm xử lý
            end_time = time.time()  # Kết thúc đo thời gian
            processing_time = round(end_time - start_time, 3)  # Tính thời gian xử lý (giây)
        except Exception as e:
            # Xử lý bất kỳ lỗi nào xảy ra trong quá trình chỉnh sửa văn bản
            error_message = f"Đã xảy ra lỗi khi xử lý văn bản: {str(e)}"
            result = None
            processing_time = None

    return render_template("index.html", text=text, result=result, processing_time=processing_time, error_message=error_message)


# Chạy server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
