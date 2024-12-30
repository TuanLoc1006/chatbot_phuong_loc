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



# import matplotlib.pyplot as plt
"""
Đọc dữ liệu từ file corpus.txt, mỗi dòng trong file đại diện cho một câu hoặc đoạn văn bản.
Chi tiết:
corpus_file_path: Đường dẫn đến file corpus.
Sử dụng with open(...) để mở file trong chế độ đọc với mã hóa UTF-8.
Duyệt từng dòng trong file, loại bỏ khoảng trắng đầu và cuối bằng strip(), sau đó lưu vào danh sách data.
In ra số lượng câu dữ liệu đã đọc.
"""
# pip install -U scikit-learn
corpus_file_path = './corpus.txt'
# Đọc dữ liệu từ file corpus.txt
with open(corpus_file_path, 'r', encoding='utf-8') as f:
    data = [line.strip() for line in f]  # Sử dụng strip() để loại bỏ khoảng trắng hoặc ký tự xuống dòng
print('Số dòng dữ liệu trong file:', len(data))

"""
vowel: Danh sách các nguyên âm tiếng Việt bao gồm cả chữ hoa và chữ thường, cũng như các dấu phụ.
full_letters: Kết hợp vowel với danh sách các phụ âm tiếng Việt.
"""

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


"""
region: Các biến thể của các nguyên âm có dấu.
region2: Các biến thể của phụ âm hoặc cụm phụ âm.
"""
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

"""
Mục đích: Thêm các từ viết tắt vào câu để tăng cường dữ liệu.
Chi tiết:
Tạo một số ngẫu nhiên random từ 0 đến 1.
Nếu random > 0.5, duyệt qua các từ trong acronym và thay thế chúng bằng dạng viết tắt với xác suất 50%.
Nếu random <= 0.5, trả về câu gốc không thay đổi.
"""
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


"""
Mục đích: Thêm nhiễu vào câu để tạo ra dữ liệu tăng cường, giúp mô hình học cách khắc phục các lỗi phổ biến.
Chi tiết:
Gọi hàm _teen_code để có thể thêm từ viết tắt.
Duyệt từng ký tự trong câu:
Nếu ký tự không phải là chữ cái (full_letters), giữ nguyên.
Với xác suất 94%, giữ nguyên ký tự.
Với xác suất 3.5%, thực hiện các thay đổi dựa trên từ điển typo và region.
Với xác suất 5.5%, có thể thay đổi phụ âm dựa trên region2 hoặc đảo chữ cái.
Hàm này sử dụng nhiều lớp ngẫu nhiên để quyết định cách thay đổi ký tự, tạo ra sự đa dạng trong dữ liệu.
"""
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


"""
Mục đích: Xử lý văn bản để tách thành các cụm từ (phrases) phù hợp để huấn luyện mô hình.
Chi tiết:
Duyệt qua từng câu trong data.
Thay thế hoặc loại bỏ các ký tự không thuộc alphabet:
Sử dụng unidecode để chuyển các ký tự không thuộc alphabet thành dạng không dấu hoặc loại bỏ chúng.
Sử dụng biểu thức chính quy re.findall(r'\w[\w\s]+', text) để tìm các cụm từ chứa các ký tự chữ cái và khoảng trắng.
Loại bỏ các cụm từ có ít hơn 2 từ.
In ra số lượng cụm từ sau khi xử lý.
"""

phrases = []
for text in data:
    # Thay thế hoặc xóa bỏ các ký tự thừa
    for c in set(text):
        if re.match('\w', c) and c not in alphabet:
            uc = unidecode(c)
            if re.match('\w', uc) and uc not in alphabet:
                text = re.sub(c, '', text)
            else:
                text = re.sub(c, uc, text)
    phrases += re.findall(r'\w[\w\s]+', text)

phrases = [p.strip() for p in phrases if len(p.split()) > 1]
print("số đoạn: ",len(phrases))

"""
Mục đích: Tạo các n-gram từ các cụm từ để tăng cường dữ liệu huấn luyện.
Chi tiết:
NGRAM = 5: Số lượng từ trong mỗi n-gram.
MAXLEN = 39: Độ dài tối đa của một n-gram.
Duyệt qua từng cụm từ trong phrases:
Nếu số từ trong cụm từ lớn hơn hoặc bằng NGRAM, tạo tất cả các n-gram có độ dài NGRAM và độ dài không vượt quá MAXLEN.
Nếu số từ nhỏ hơn NGRAM nhưng độ dài không vượt quá MAXLEN, thêm cụm từ đó vào list_ngrams.
Loại bỏ các n-gram trùng lặp bằng cách chuyển list_ngrams thành set rồi lại chuyển ngược thành list.
In ra số lượng n-gram sau khi xử lý.
"""
NGRAM = 5
MAXLEN = 39
list_ngrams = []
for p in phrases:
    list_p = p.split()
    if len(list_p) >= NGRAM:
        for ngr in ngrams(p.split(), NGRAM):
            if len(' '.join(ngr)) <= MAXLEN:
                list_ngrams.append(' '.join(ngr))
    elif len(' '.join(list_p)) <= MAXLEN:
        list_ngrams.append(' '.join(list_p))

list_ngrams = list(set(list_ngrams))
print(len(list_ngrams))

"""
Mục đích: Chuyển đổi văn bản thành định dạng one-hot encoding để sử dụng trong mô hình học máy.
Chi tiết:
Tạo một mảng numpy x có kích thước (MAXLEN, len(alphabet)), khởi tạo bằng 0.
Duyệt qua từng ký tự trong văn bản (tối đa MAXLEN ký tự):
Đánh dấu vị trí của ký tự đó trong alphabet bằng cách đặt giá trị 1 tại vị trí tương ứng.
Nếu số ký tự trong văn bản ít hơn MAXLEN, điền vào các vị trí còn lại bằng cách đặt 1 ở cột đầu tiên ('\x00').
Trả về mảng mã hóa x.
"""
def _encoder_data(text):
  x = np.zeros((MAXLEN, len(alphabet)))
  for i, c in enumerate(text[:MAXLEN]):
    x[i, alphabet.index(c)] = 1
  if i <  MAXLEN - 1:
    for j in range(i+1, MAXLEN):
      x[j, 0] = 1
  return x

"""
Mục đích: Chuyển đổi định dạng one-hot encoding trở lại thành văn bản.
Chi tiết:
Tìm chỉ số có giá trị lớn nhất (1) trong mỗi hàng của mảng x bằng argmax.
Chuyển các chỉ số này thành ký tự tương ứng trong alphabet.
Nối các ký tự lại thành một chuỗi văn bản.
"""
def _decoder_data(x):
  x = x.argmax(axis=-1)
  return ''.join(alphabet[i] for i in x)

"""
Xây dựng một mô hình mạng nơ-ron sâu để xử lý chuỗi ký tự và học cách chỉnh sửa văn bản.
Chi tiết:
Encoder:
Sử dụng một lớp LSTM với 256 đơn vị.
input_shape=(MAXLEN, len(alphabet)): Mỗi đầu vào có độ dài MAXLEN và mỗi ký tự được mã hóa bằng one-hot với kích thước len(alphabet).
return_sequences=True: Trả về toàn bộ chuỗi đầu ra để có thể tiếp nối với decoder.
Decoder:
Sử dụng một lớp LSTM hai chiều (Bidirectional) với 256 đơn vị.
dropout=0.2: Áp dụng dropout để giảm overfitting.
return_sequences=True: Trả về toàn bộ chuỗi đầu ra.
Các lớp tiếp theo:
TimeDistributed(Dense(256)): Áp dụng một lớp Dense cho mỗi thời điểm trong chuỗi.
Activation('relu'): Hàm kích hoạt ReLU.
TimeDistributed(Dense(len(alphabet))): Lớp Dense để chuyển đổi lại thành kích thước của alphabet.
Activation('softmax'): Hàm kích hoạt Softmax để xác định xác suất của mỗi ký tự trong alphabet.
Biên dịch mô hình:
Sử dụng hàm mất mát categorical_crossentropy phù hợp với bài toán phân loại đa lớp.
Sử dụng optimizer Adam với tốc độ học 0.001.
Theo dõi chỉ số accuracy.
"""

encoder = LSTM(256, input_shape=(MAXLEN, len(alphabet)), return_sequences=True)
decoder = Bidirectional(LSTM(256,  return_sequences=True, dropout=0.2))

model = Sequential()
model.add(encoder)
model.add(decoder)
model.add(TimeDistributed(Dense(256)))
model.add(Activation('relu'))
model.add(TimeDistributed(Dense(len(alphabet))))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])
# model.summary()
# from tensorflow.keras.utils import plot_model
# plot_model(model, to_file='model.png')

"""
Mục đích: Chia dữ liệu thành hai tập: huấn luyện (80%) và kiểm tra (20%) để đánh giá hiệu suất mô hình.
Chi tiết:
Sử dụng train_test_split từ thư viện sklearn để chia dữ liệu.
test_size=0.2: 20% dữ liệu dành cho tập kiểm tra.
random_state=42: Đặt seed để đảm bảo kết quả chia dữ liệu có thể tái lập.
"""
from sklearn.model_selection import train_test_split
train_data, valid_data = train_test_split(list_ngrams, test_size=0.2, random_state=42)

"""
Mục đích: Tạo các batch dữ liệu để huấn luyện mô hình mà không cần tải toàn bộ dữ liệu vào bộ nhớ cùng một lúc.
Chi tiết:
data: Dữ liệu nguồn (tập huấn luyện hoặc kiểm tra).
batch_size: Kích thước của mỗi batch.
Vòng lặp vô hạn (while True) để liên tục cung cấp dữ liệu cho mô hình.
Trong mỗi batch:
y: Đối tượng mục tiêu (dữ liệu gốc được mã hóa).
x: Đầu vào (dữ liệu đã được thêm nhiễu và mã hóa).
Nếu đạt đến cuối dữ liệu, quay lại đầu danh sách.
"""
# # Chia tách dữ liệu để tránh tràn RAM
BATCH_SIZE = 512
EPOCHS = 10
def _generate_data(data, batch_size):
    current_index = 0
    while True:
        x, y = [], []
        for i in range(batch_size):
            y.append(_encoder_data(data[current_index]))
            x.append(_encoder_data(_add_noise(data[current_index])))
            current_index += 1
            if current_index > len(data) - 1:
                current_index = 0
        yield (np.array(x), np.array(y))

"""
Mục đích: Tạo các generators để cung cấp dữ liệu cho quá trình huấn luyện và kiểm tra.
Chi tiết:
BATCH_SIZE = 512: Mỗi batch chứa 512 mẫu dữ liệu.
EPOCHS = 10: Số lần lặp qua toàn bộ dữ liệu huấn luyện.
"""
train_generator = _generate_data(train_data, batch_size = BATCH_SIZE)
validation_generator = _generate_data(valid_data, batch_size = BATCH_SIZE)

# train model
# H = model.fit(
#     train_generator, epochs = EPOCHS,
#     steps_per_epoch=len(train_data) // BATCH_SIZE,
#     validation_data=validation_generator,
#     validation_steps=len(valid_data) // BATCH_SIZE
# )
# model.save('./my_model/model_22_9_2024.h5')

# Define the path to the model (adjust this path based on your environment)
model_path = './model_31_10_2024.h5'
# model_path = './model_4_12_2024.h5'
model = load_model(model_path)

"""
Mục đích: Đảm bảo rằng các hàm mã hóa và giải mã dữ liệu được định nghĩa lại để sử dụng trong phần dự đoán.
Chi tiết: Các hàm này giống như đã được định nghĩa trước đó để chuẩn hóa dữ liệu đầu vào và giải mã kết quả dự đoán.
"""
NGRAM = 5
MAXLEN = 39


"""
Mục đích: Tạo các n-gram từ câu văn sử dụng thư viện NLTK.
Chi tiết:
Sử dụng nltk.ngrams để tạo các n-gram từ danh sách từ.
Chỉ thêm các n-gram có độ dài không vượt quá maxlen.
Nếu số từ ít hơn n, thêm toàn bộ câu như một n-gram duy nhất.
"""
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


"""
Mục đích: Dự đoán sửa lỗi cho một n-gram cụ thể.
Chi tiết:
Kết hợp các từ trong n-gram thành một chuỗi văn bản.
Mã hóa chuỗi văn bản và đưa vào mô hình để dự đoán.
Giải mã kết quả dự đoán và loại bỏ ký tự padding ('\x00').
"""

# def _guess(ngram):
#     text = " ".join(ngram)
#     preds = model.predict(np.array([_encoder_data(text)]))

#     return _decoder_data(preds[0]).strip('\x00')

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


# Ví dụ trong hàm _guess
# def _guess(ngram):
#     text = " ".join(ngram)
    
#     # Đảm bảo rằng dấu '/' và '-' không bị loại bỏ trong quá trình xử lý
#     text = re.sub(r"[^\w\s/-]", '', text) 

"""
Mục đích: Thêm lại các dấu câu vào văn bản đã được sửa lỗi.
Chi tiết:
Duyệt qua từng từ trong văn bản gốc text để xác định vị trí các dấu câu ở đầu và cuối từ.
Lưu các dấu câu này vào list_punctuation với chỉ số vị trí từ.
Sau đó, kết hợp các từ đã được sửa lỗi corrected_text với các dấu câu tương ứng từ list_punctuation.
"""

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


"""
Mục đích: Chỉnh sửa văn bản nhập vào bằng cách sử dụng mô hình đã huấn luyện để sửa lỗi.
Chi tiết:
Bước 1: Loại bỏ các ký tự đặc biệt không thuộc alphabet.
Bước 2: Tạo các n-gram từ văn bản đã được làm sạch.
Bước 3: Dự đoán sửa lỗi cho từng n-gram bằng hàm _guess.
Bước 4: Sử dụng Counter để xác định từ được dự đoán phổ biến nhất tại mỗi vị trí trong câu.
Bước 5: Kết hợp các từ đã được sửa lỗi thành văn bản cuối cùng và thêm lại các dấu câu.
"""

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

"""sai"""

# def _correct(text):
#     # Xóa các ký tự đặc biệt
#     new_text = re.sub(r"[^" + ''.join(alphabet) + ']', '', text)
#     print("Step 1: After removing special characters ->", new_text)
    
#     # Tạo n-grams từ văn bản
#     ngrams = list(_nltk_ngrams(new_text, NGRAM, MAXLEN))
#     print("Step 2: Generated n-grams ->", ngrams)
    
#     # Dự đoán từ hoặc cụm từ từ các n-grams
#     guessed_ngrams = list(_guess(ngram) for ngram in ngrams)
#     print("Step 3: After guessing n-grams ->", guessed_ngrams)
    
#     # Khởi tạo danh sách các Counter để đếm từ
#     candidates = [Counter() for _ in range(len(guessed_ngrams) + NGRAM - 1)]
    
#     # Cập nhật danh sách ứng viên
#     for nid, ngram in enumerate(guessed_ngrams):
#         for wid, word in enumerate(re.split(r'\s', ngram)):
#             index = nid + wid
#             if index < len(candidates):
#                 candidates[index].update([word])
#             else:
#                 candidates.append(Counter([word]))
#     print("Step 4: Candidates ->", candidates)
    
#     # Tạo lại văn bản đã sửa từ các từ phổ biến nhất
#     corrected_text = ' '.join(c.most_common(1)[0][0] for c in candidates if c)
#     print("Step 5: Corrected text without punctuation ->", corrected_text)
    
#     # Thêm lại dấu câu
#     result = _add_punctuation(text, corrected_text)
#     print("Step 6: Final corrected text ->", result)
    
#     return result




# while(True):
#     text = input("nhập nội dung/nhấn 'q' để thoát:\n")
#     if text == 'q':
#         break
#     print("câu sai: "+text)
#     print(_correct(text))
#     result = _correct(text)
#     text = re.sub(r"[^" + ''.join(alphabet) + ']', '', text)
#     list_text = text.split()

#     result = re.sub(r"[^" + ''.join(alphabet) + ']', '', result)
#     list_result = result.split()

    # hien thi cac tu da sua sai
    # Hiển thị những từ đã sửa
    # corrected_word = [(list_text[i], list_result[i]) for i in range(len(list_text)) if list_text[i] != list_result[i]]
   
    # print(corrected_word)















# import re

# while True:
#     text = input("Nhập nội dung/nhấn 'q' để thoát:\n")
#     if text == 'q':
#         break
    
#     result = _correct(text)
#     print(result)
#     # Loại bỏ các ký tự không thuộc bảng chữ cái
#     text = re.sub(r"[^" + ''.join(alphabet) + ']', '', text)
#     list_text = text.split()

#     result = re.sub(r"[^" + ''.join(alphabet) + ']', '', result)
#     list_result = result.split()

#     # Kiểm tra độ dài của list_text và list_result
#     if len(list_text) != len(list_result):
#         print("Dữ liệu sai chính tả nhiều quá")
#         continue
    
#     # Hiển thị những từ đã sửa
#     corrected_word = [(list_text[i], list_result[i]) for i in range(len(list_text)) if list_text[i] != list_result[i]]
    
#     print("Các từ đã sửa:", corrected_word)



# chong thowfi đại soos óha hiện lay, văn bản đánh máy ddax daafn thay thế vawn barn viết tay bởi sự thuận tieejn của nó. Kefm theo đó, lỗi chính tả xuất hiện trog lúc soạn tharo nà điều ko thể tránh khỏi. Một số ný do gaay nên lỗi chính tả là: lỗi gõ bàn phím, khác biệt vùng miền, viết tắt,.. Điều này đã dấn đến nhu cầu hệ thống giúp phát hiện và sửa lỗi chính tả trong văn bản tiếng Việt.




# -------------------------demo-------------------------

import numpy as np
import re
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score
import nltk

sentences_with_errors = [
    "tooi iu tiếng Việt",
    "học phis ngafnh duowjcj bao nhiêu",
    "truownfgf có chisnh sách hoox trợ gì cho sinh vieen",
    "trường cos các hoajt động ngoại khoá ko? ",
    "địa chir của truownfgw ở đaau",

    "Điểm chuẩn đầu vafo các ngafnh của truowngf là bao nhiêu?",
    "Khu vực xung quanh truowngwff có nhieefu chỗ ở trọ không?",
    "Em cos thể tham gia các caau lạc bộ sinh viên nào ở truwongfw ?",
    "trường cos các hoajt động ngoại khoá khoong? ",
    "Sinh viên cos thể đi lafm thêm khi học khoong? ",
    "Truownfgw có tổ chức các chuongw trình trao đỏi sinh vieen khoong?",
    "Cow hội thực tâjp ở các bệnh viện lớn thees nafo?",
    "Có quán ăn hay căng tin troong khuoon vien trường không?",
    "Casc hoajt động thể thao cho sinh vien nhuw thế nafo?",
    "Trường co trung taam nghiên cuuwss y hojc khoong?",
    "Sinh viên có dduowjcw tham gia vào các dự án nghiên cuuwss không",
    "Trường có chương trình hoox trowj khởi nghiệp cho sinh vieen không?",
    "Lafm sao để tham gia các dự asn nghiên cứu của giảng viên?",
    "Có thể xin quyx hỗ trowj nghieen cứu từ trường khoong?",
    "Trường cos hơp tasc với beejnh viện nào để nghieen cứu khoong?",
    "Các công trifnh nghiên cứu của sinh vieen có đươjc công boos không?",
    "Sinh viên có theer thực hiện nghiên cuws tuwf năm nao?",
    "Các thiêt bij phong this nghiệm của trường có hieejn đại khoong?",
    "Trường có tổ chuwsc hội thao nghieen cuuwss hàng nawm khoong?",
    "Thuw viện của trường có mở cả cuoosi tuần khoong?",
    "Cos phofng tuwj học cho sinh vieen khoong?",
    "Trường có heej thống e-learning không?",
    "Casc phòng thí nghiệm có đủ thieest bị cho sinh viên khoong?",
    "trường có wifi miễn phis không?",
    "Khuoon viên trường có rộng khoong vaajy?",
    "Có chôx để xe miễn  phí cho sinh viên không?",
    "Trường cos các phofng họp nhosm không?",
    "Cawng tin trong trường co basn thức ăn gias sinh viên không?",
    "Trường có hệ thoonsgs kí túc xá hieẹn đại không?",
    "Học phis ngafnh Y đa khoa là bao nhieeu?",
    "Trường có chính sasch hỗ trợ hojc phí cho sinh viên khó khawn không?",
    "Hoc phis có tăng theo từng nawm không?",
    "Sinh viên có đươc giarm học phi nếu có hoàn carnh khó khawn không?",
    "Trường có học boorng khuyến khích học tập không?",
    "hojc bổng cura trường có dành cho sinh vieen quoosc tees không?",
    "Học phí nganh Dược so với casc trường khác cos cao khoong vaajy?",
    "Trường có yêu caafu đosng hojc phi trước khi nhập hojc khoong?",
    "Thowfi gian đóng hojc phi là khi nào vâjy?",
    "Truwongf cos hỗ trowj sinh vien vay vốn hojc tập khoong?",
    "trường có tổ chuwcs thực tập cho sinh vien khong?",
    "Sinh vien thực tập owr ddaau khi hojc ngafnh Y khoa?",
    "Có theer thuwjc tập owr các bệnh vieejn ngoafi tỉnh khong?",
    "trường có hợp tác với nhuwngx bệnh viện nào cho sinh vien thuwjc tập?",
    "Sinh vieen có theer thuwjc tap owr bệnh viện nafo trong nawm cuoosi?",
    "Cow hooji việc lafm sau khi tốt nghiêjp ngafnh Y thế nào?",
    "trường cos chuongw trifnh ho trowj viejc làm sau tốt nghiệp khong?",
    "Các bệnh viện có thuowngfw xuyeen tuyển dụng sinh vieen trường khong?",
    "Sinh vien co theer xin việc lafm theem trong nganh khoong?",
    "Lafm sao deer tifm được việc lafm sau khi ra truowngf?",
    "trường cos chương trinh hojc quốc tees khoong?",
    "Lafm sao để ddawng kys hojc trao đổi quốc tế",
    "Điều kiện để được tham gia chương trình trao ddooori sinh vieen là gì?",
    "Các nước nafo trường có howjp tác trao ddoori sinh viên?",
    "trường có lieen kết với trường ddaji hojc quốc tế nafo khoong?",
    "Sinh viên có thể du hojc ngawnss hajn trong thời gian học tại trường khoong?",
    "Các chuongw trình trao đổi sinh vieen thường keso dài bao laau?",
    "Truowngwfw có hỗ trợ sinh viên hojc tiếng Anh không?",
    "Sinh vieen trao đổi quoosc tees có được hỗ trowj hojc phí không?",
    "Đooji ngux giảng viên của trường có kinh nghieejm như thế nào?",
    "Giarng viên có banwgfw cấp quốc tees khoong?",
    "Các giarng viên có thường xuyên cập nhật kiến thuwsc y hojc mới khong?",
    "Có nhieefu giảng viên laf basc sĩ khoong?",
    "trường cos tổ chức các khóa học ngắn hajn khong?",
    "Có giảng viên nuowscs ngoài tham gia giarng dajy học không?",
    "Các giảng vieen có hỗ trowj sinh viên ngoafi giowf học không?",
    "Sinh viên cos theer học với các giáo suw nổi tiếng khong?",
    "Trươfng có chất luowngjw  giarng dajy tốt không?",
    "Có cow hội học từ giảng viên quốc tees không?",
    "Truownfgw có tổ chuwsc trương trình khám chữa bệnh mễn phí không?",
    "Sinh viên cos ddduowjcd tham gia các chương trình từ thiện y tế khoong?",
    "truownfgf có liên kêts với các tổ chức y têe quooc tế không?",
    "Các dưj án y tế công cộng thuofng tập trung vào lixnh vưjc gì?",
    "Sinh vieen có được thma gia vafo các duwjw án y tế công cộng không?",
    "Có chuongw trifnh tifnh nguyện y tế khoong?",
    "truowngfw có hopwjw tác với các tổ chuwsc phi chính phủ ko?",
    "Có các chương trình hỗ trowj sức khoẻ cộng đoofng không?",
    "Các dưj án y tế cộng đồng của trường có tâmd ảnh hưởng lớn ko?",
    "Truofng đã đat được nhung thafnh tựu nào nổi bật?",
    "Đaji hocj Y Dược Cần Thơ co nằm trong bảng xeesp hạng nào khoong?",
    "trường có truyeefn thống nghiên cuuws y khoa không?",
    "Các cựu sinh vieen cura truownfgw có thành tựu gì?",
    "Uy tín của trường so vói các truownfgf y dược khác thế nafo?",
    "truongwf có từng toor chức cac sự kieejn quoosc tees ko?",
    "Casc giáo suw nổi tiếng nào ddang giảng dạy tại trương?",
    "Truownfgw co mối quan hệ howjp tasc với các cơ quan y tế nào?",
    "Đooj uy tisn của trường trong khu vuwjc nhuw thế nào?",
    "trường có tuyeern sinh liên thoong khoong",
    "Bao nhieeu ngành ddang được đafo tajo tại trường",
    "trường cos học boorng cho sinh vieen khos khăn không",
    "trường cos phong masy tính khoong",
    "Học phí cos tawng theo từng nawm không",
    "Sinh viên cos ddược học tieesng Anh ko",
    "Trương có cho phesp sinh vieen xin học laji các moon không",
    "điểm chuaanrr của ngành y",
    "trường có những hoạt động ngoaji khoas nafo cho sinh vieen vaayja",
    "Truownfgf có tổ chức các buoori tharo luận vê y đức ko?",
    "Sinh vieen có được tham gia casc khoas hojc ngoại khóa về y ddức ko?",
    "Trường cos trung taam tư vấn taam lý cho sinh vieen ko?",
    "Sinh viên có ddược hỗ trợ về suwsc khỏe tinh thần ko?",
    "Có chuongw trình hỗ trợ sinh viên gặp khos khăn về tài chính ko?",
    "Trương có cung cấp các dịch vụ y tees mieenxx phí sv ko?",
    "Sinh vieen có thể tham gia casc khóa học về quan lys cawng thẳng ko?",
    "Trươfng có chương trifnh giúp sv cân bawfng giữa học tập và cuoojc sống không?",
    "Có chương trình nào hỗ trợ sinh viên gặp kho khăn trong học tập ko?",
    "Trường có ddoijj ngũ chuyeen viên tâm lý hỗ trowj sinh viên không?",
    "Casc dịch vụ hoox trợ sinh vien có hoạt động trong suoost kỳ nghỉ ko?",


]

# Dự đoán các câu đã chỉnh sửa bằng mô hình
predicted_sentences = [_correct(sentence) for sentence in sentences_with_errors]


# Kiểm tra kết quả
for i, (original, predicted) in enumerate(zip(sentences_with_errors, predicted_sentences)):
    
    print(f"Câu {i+1}: Câu hỏi ban đầu: {original}")
    print(f"        Câu đã sửa lỗi: {predicted}")
   














# câu 15 -> 133
# "Điểm chuẩn đầu vafo các ngafnh của truowngf là bao nhiêu?",
# "Khu vực xung quanh truowngwff có nhieefu chỗ ở trọ không?",
# "Em cos thể tham gia các caau lạc bộ sinh viên nào ở truwongfw ?",
# "trường cos các hoajt động ngoại khoá khoong? ",
# "Sinh viên cos thể đi lafm thêm khi học khoong? ",
# "Truownfgw có tổ chức các chuongw trình trao đỏi sinh vieen khoong?",
# "Cow hội thực tâjp ở các bệnh viện lớn thees nafo?",
# "Có quán ăn hay căng tin troong khuoon vien trường không?",
# "Casc hoajt động thể thao cho sinh vien nhuw thế nafo?",
#  "Trường co trung taam nghiên cuuwss y hojc khoong?",
# "Sinh viên có dduowjcw tham gia vào các dự án nghiên cuuwss không",
# "Trường có chương trình hoox trowj khởi nghiệp cho sinh vieen không?",
# "Lafm sao để tham gia các dự asn nghiên cứu của giảng viên?",
# "Có thể xin quyx hỗ trowj nghieen cứu từ trường khoong?",
# "Trường cos hơp tasc với beejnh viện nào để nghieen cứu khoong?",
# "Các công trifnh nghiên cứu của sinh vieen có đươjc công boos không?",
# "Sinh viên có theer thực hiện nghiên cuws tuwf năm nao?",
# "Các thiêt bij phong this nghiệm của trường có hieejn đại khoong?",
# "Trường có tổ chuwsc hội thao nghieen cuuwss hàng nawm khoong?",
# "Thuw viện của trường có mở cả cuoosi tuần khoong?",
# "Cos phofng tuwj học cho sinh vieen khoong?",
# "Trường có heej thống e-learning không?",
# "Casc phòng thí nghiệm có đủ thieest bị cho sinh viên khoong?",
# "trường có wifi miễn phis không?",
# "Khuoon viên trường có rộng khoong vaajy?",
# "Có chôx để xe miễn  phí cho sinh viên không?",
# "Trường cos các phofng họp nhosm không?",
# "Cawng tin trong trường co basn thức ăn gias sinh viên không?",
# "Trường có hệ thoonsgs kí túc xá hieẹn đại không?",
# "Học phis ngafnh Y đa khoa là bao nhieeu?",
# "Trường có chính sasch hỗ trợ hojc phí cho sinh viên khó khawn không?",
# "Hoc phis có tăng theo từng nawm không?",
# "Sinh viên có đươc giarm học phi nếu có hoàn carnh khó khawn không?",
# "Trường có học boorng khuyến khích học tập không?",
# "hojc bổng cura trường có dành cho sinh vieen quoosc tees không?",
# "Học phí nganh Dược so với casc trường khác cos cao khoong vaajy?",
# "Trường có yêu caafu đosng hojc phi trước khi nhập hojc khoong?",
# "Thowfi gian đóng hojc phi là khi nào vâjy?",
# "Truwongf cos hỗ trowj sinh vien vay vốn hojc tập khoong?",
# "trường có tổ chuwcs thực tập cho sinh vien khong?",
# "Sinh vien thực tập owr ddaau khi hojc ngafnh Y khoa?",
# "Có theer thuwjc tập owr các bệnh vieejn ngoafi tỉnh khong?",
# "trường có hợp tác với nhuwngx bệnh viện nào cho sinh vien thuwjc tập?",
# "Sinh vieen có theer thuwjc tap owr bệnh viện nafo trong nawm cuoosi?",
# "Cow hooji việc lafm sau khi tốt nghiêjp ngafnh Y thế nào?",
# "trường cos chuongw trifnh ho trowj viejc làm sau tốt nghiệp khong?",
# "Các bệnh viện có thuowngfw xuyeen tuyển dụng sinh vieen trường khong?",
# "Sinh vien co theer xin việc lafm theem trong nganh khoong?",
# "Lafm sao deer tifm được việc lafm sau khi ra truowngf?",
# "trường cos chương trinh hojc quốc tees khoong?",
# "Lafm sao để ddawng kys hojc trao đổi quốc tế",
# "Điều kiện để được tham gia chương trình trao ddooori sinh vieen là gì?",
# "Các nước nafo trường có howjp tác trao ddoori sinh viên?",
# "trường có lieen kết với trường ddaji hojc quốc tế nafo khoong?",
# "Sinh viên có thể du hojc ngawnss hajn trong thời gian học tại trường khoong?",
# "Các chuongw trình trao đổi sinh vieen thường keso dài bao laau?",
# "Truowngwfw có hỗ trợ sinh viên hojc tiếng Anh không?",
# "Sinh vieen trao đổi quoosc tees có được hỗ trowj hojc phí không?",
# "Đooji ngux giảng viên của trường có kinh nghieejm như thế nào?",
# "Giarng viên có banwgfw cấp quốc tees khoong?",
# "Các giarng viên có thường xuyên cập nhật kiến thuwsc y hojc mới khong?",
# "Có nhieefu giảng viên laf basc sĩ khoong?",
# "trường cos tổ chức các khóa học ngắn hajn khong?",
# "Có giảng viên nuowscs ngoài tham gia giarng dajy học không?",
# "Các giảng vieen có hỗ trowj sinh viên ngoafi giowf học không?",
# "Sinh viên cos theer học với các giáo suw nổi tiếng khong?",
# "Trươfng có chất luowngjw  giarng dajy tốt không?",
# "Có cow hội học từ giảng viên quốc tees không?",
# "Truownfgw có tổ chuwsc trương trình khám chữa bệnh mễn phí không?",
# "Sinh viên cos ddduowjcd tham gia các chương trình từ thiện y tế khoong?",
# "truownfgf có liên kêts với các tổ chức y têe quooc tế không?",
# "Các dưj án y tế công cộng thuofng tập trung vào lixnh vưjc gì?",
# "Sinh vieen có được thma gia vafo các duwjw án y tế công cộng không?",
# "Có chuongw trifnh tifnh nguyện y tế khoong?",
# "truowngfw có hopwjw tác với các tổ chuwsc phi chính phủ ko?",
# "Có các chương trình hỗ trowj sức khoẻ cộng đoofng không?",
# "Các dưj án y tế cộng đồng của trường có tâmd ảnh hưởng lớn ko?",
# "Truofng đã đat được nhung thafnh tựu nào nổi bật?",
# "Đaji hocj Y Dược Cần Thơ co nằm trong bảng xeesp hạng nào khoong?",
# "trường có truyeefn thống nghiên cuuws y khoa không?",
# "Các cựu sinh vieen cura truownfgw có thành tựu gì?",
# "Uy tín của trường so vói các truownfgf y dược khác thế nafo?",
# "truongwf có từng toor chức cac sự kieejn quoosc tees ko?",
# "Casc giáo suw nổi tiếng nào ddang giảng dạy tại trương?",
# "Truownfgw co mối quan hệ howjp tasc với các cơ quan y tế nào?",
# "Đooj uy tisn của trường trong khu vuwjc nhuw thế nào?",
# "trường có tuyeern sinh liên thoong khoong",
# "Bao nhieeu ngành ddang được đafo tajo tại trường",
# "trường cos học boorng cho sinh vieen khos khăn không",
# "trường cos phong masy tính khoong",
# "Học phí cos tawng theo từng nawm không",
# "Sinh viên cos ddược học tieesng Anh ko",
# "Trương có cho phesp sinh vieen xin học laji các moon không",
# "điểm chuaanrr của ngành y",
# "trường có những hoạt động ngoaji khoas nafo cho sinh vieen vaayja",




# câu 162 -> 172
# "Truownfgf có tổ chức các buoori tharo luận vê y đức ko?",
# "Sinh vieen có được tham gia casc khoas hojc ngoại khóa về y ddức ko?",
# "Trường cos trung taam tư vấn taam lý cho sinh vieen ko?",
# "Sinh viên có ddược hỗ trợ về suwsc khỏe tinh thần ko?",
# "Có chuongw trình hỗ trợ sinh viên gặp khos khăn về tài chính ko?",
# "Trương có cung cấp các dịch vụ y tees mieenxx phí sv ko?",
# "Sinh vieen có thể tham gia casc khóa học về quan lys cawng thẳng ko?",
# "Trươfng có chương trifnh giúp sv cân bawfng giữa học tập và cuoojc sống không?",
# "Có chương trình nào hỗ trợ sinh viên gặp kho khăn trong học tập ko?",
# "Trường có ddoijj ngũ chuyeen viên tâm lý hỗ trowj sinh viên không?",
# "Casc dịch vụ hoox trợ sinh vien có hoạt động trong suoost kỳ nghỉ ko?",
    


    
# wifi
# miễn
    # Ddooidjd
    # thiện

    # đào tạo
    # chuẩn
    # điểm
    # những
    # miễn

while(True):
    text = input("nhập nội dung/nhấn 'q' để thoát:\n")
    if text == 'q':
        break
    print("câu sai: "+text)
    print(_correct(text))
    result = _correct(text)
    text = re.sub(r"[^" + ''.join(alphabet) + ']', '', text)
    list_text = text.split()

    result = re.sub(r"[^" + ''.join(alphabet) + ']', '', result)
    list_result = result.split()

    # hien thi cac tu da sua sai
    # Hiển thị những từ đã sửa
    corrected_word = [(list_text[i], list_result[i]) for i in range(len(list_text)) if list_text[i] != list_result[i]]
   
    print(corrected_word)