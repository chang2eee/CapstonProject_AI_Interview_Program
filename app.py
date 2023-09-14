from flask import Flask, session, render_template, redirect, request, url_for, Response
from flaskext.mysql import MySQL
import voice
import requests
import json

import cv2
import numpy as np
from keras.models import load_model
from statistics import mode
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

app = Flask(__name__)

app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = 'jaewoo55'
app.config['MYSQL_DATABASE_DB'] = 'test_db'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'
app.secret_key = "ABCDEFG"

mysql = MySQL(app)
mysql.init_app(app)

# 매개 변수를 위한 데이터 및 이미지 로드
emotion_model_path = './models/emotion_model.hdf5'
emotion_labels = get_labels('fer2013')
emotion_offsets = (20, 40)

# 학습된 모델 로드
face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
emotion_classifier = load_model(emotion_model_path)

emotion_target_size = emotion_classifier.input_shape[1:3]
emotion_window = []
frame_window = 10  # 프레임 창 크기: 10

def detect_emotion(frame):
    global emotion_window

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5,
                                          minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for face_coordinates in faces:
        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]

        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)

        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
        else:
            color = emotion_probability * np.asarray((0, 255, 0))

        color = color.astype(int)
        color = color.tolist()

        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_mode,
                  color, 0, -45, 1, 1)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    return bgr_image


# 비디오 스트리밍 함수
def video_stream():
    # cv2.VideoCapture(0)했는데 안돼서 아래와 같이 수정함
    # 0은 내장 웹캠, 그 외의 다른 숫자는 외부 연결 캠
    # FFMPEG 라이브러리 사용 -> 해결 완료!
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        ret, frame = cap.read()  
        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame = detect_emotion(frame)

        # 이미지 스트리밍을 위한 바이트 변환
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()
    
    
# 비디오 스트리밍 라우트
@app.route('/video_feed')
def video_feed():
    return Response(video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    
    
@app.route('/', methods=['GET', 'POST'])
def main():
    error = None
 
    if request.method == 'POST':
        id = request.form['id']
        pw = request.form['pw']
 
        conn = mysql.connect()
        cursor = conn.cursor()
        sql = "SELECT id FROM user WHERE id = %s AND pw = %s"
        value = (id, pw)
        cursor.execute("set names utf8")
        cursor.execute(sql, value)
 
        data = cursor.fetchall()
        cursor.close()
        conn.close()
 
        for row in data:
            data = row[0]
 
        if data:
            session['login_user'] = id
            return redirect(url_for('home'))
        else:
            error = 'invalid input data detected !'
    return render_template('main.html', error = error)
 
 
@app.route('/register.html', methods=['GET', 'POST'])
def register():
    error = None
    if request.method == 'POST':
        id = request.form['regi_id']
        pw = request.form['regi_pw']
 
        conn = mysql.connect()
        cursor = conn.cursor()
 
        sql = "INSERT INTO user VALUES ('%s', '%s')" % (id, pw)
        cursor.execute(sql)
 
        data = cursor.fetchall()
 
        if not data:
            conn.commit()
            return redirect(url_for('main'))
        else:
            conn.rollback()
            return "Register Failed"
 
        cursor.close()
        conn.close()
    return render_template('register.html', error=error)
 

@app.route('/home.html', methods=['GET', 'POST'])
def home():
    error = None
    id = session['login_user']
    return render_template('home.html', error=error, name=id)


# 추가
@app.route('/logout', methods=['GET'])
def logout():
    session.pop('name',None)
    return redirect('/')


@app.route('/mypage.html', methods=['GET', 'POST'])
def mypage():
   return render_template('mypage.html')


@app.route('/shooting_page.html', methods=['GET', 'POST'])
def shooting_page():
   return render_template('shooting_page.html')


# 추가(2023.03.30)
@app.route('/test', methods=["GET", "POST"])
def test():
    return render_template('test.html')


# 추가(2023.05.08)
@app.route('/test2', methods=["GET", "POST"])
def test2():
    text_file = 'myInterview.txt'
    result = voice.startRecord(text_file)
    text = result["text"]
    runningTime = result["runningTime"]
    # 추가(2023.05.20)
    data, count = voice.fileopen(text_file)
    # 추가한 매개변수(wordCountResult, wordLen, wordSpeed)
    wordCountResult = voice.count_character(data) + count-1
    wordSpeed = voice.compareSpeakingVel(wordCountResult, runningTime, result)
    
    ##########################################################################
   
    # 한국어 맞춤법 검사기 API URL
    CHECKER_URL = 'http://164.125.7.61/speller/results'

    # 입력 파일에서 텍스트 읽기
    with open(text_file, 'r', encoding='UTF-8') as file:
        text = file.read()

    # 개행 문자를 \r\n으로 변환
    text = text.replace('\n', '\r\n')

    # 맞춤법 검사기 API에 텍스트 전송
    response = requests.post(CHECKER_URL, data={'text1': text})

    # API 응답에서 교정된 단어 추출
    data = response.text.split('data = [', 1)[-1].rsplit('];', 1)[0]
    data = json.loads(data)
    corrected_text = text

    # 교정된 단어로 입력 텍스트 교체
    for err in data['errInfo']:
        corrected_text = corrected_text.replace(err['orgStr'], err['candWord'])

    # 교정된 텍스트를 출력 파일에 쓰기
    with open(text_file, 'w', encoding='UTF-8') as file:
        file.write(corrected_text)


    # 부산대학교에서 제공해주는 맞춤법 교정기 사용하면, 스페이스 공백 뿐 만 아닌
    # '자료구조|자로구조' 와 같이 나오는 경우 존재
    # 스페이스 공백을 기준으로 먼저 list에 정렬한 후, 해당 list의 요소를 순회하여 '|'가 존재하면 '|'를 기준으로 또 나누는 프로그램
    # 정확한 keyword 추출을 위해서 동일한 방법으로 조사를 모두 제거하는 방향으로 프로그래밍 수행 (Line 143 ~ Line 223)

    # 입력 파일 열기
    with open(text_file, 'r', encoding='UTF-8') as file:
        # 파일 내용을 문자열로 읽기
        contents = file.read()

    # 문자열을 띄어쓰기 단위로 분리하여 리스트에 저장
    word_list = contents.split()

    # 리스트 요소를 순회하면서 | 가 있는 요소를 처리
    for i, word in enumerate(word_list):

        if '|' in word:
            # '|'를 기준으로 요소를 분리하여 새로운 리스트에 저장
            sub_list = word.split('|')
            # 분리된 요소를 기존 리스트에 덮어쓰기
            word_list[i:i+1] = sub_list
        else:
            pass

        if '을' in word:
            # '을' 기준
            sub_list = word.split('을')
            word_list[i:i+1] = sub_list
        else:
            pass
        
        if '를' in word:
            # '를' 기준
            sub_list = word.split('를')
            word_list[i:i+1] = sub_list
        else:
            pass

        # '이'의 경우 : '데이터베이스'에서 이가 모두 사라지게 되어 '데터베스'로 변환
        # 사용자사전을 이용하여 다시 '데터베스'를 '데이터베이스'로 변환할 예정
        if '이' in word:
            # '이' 기준
            sub_list = word.split('이') 
            word_list[i:i+1] = sub_list
        else:
            pass

        if '가' in word:
            # '가' 기준
            sub_list = word.split('가') 
            word_list[i:i+1] = sub_list
        else:
            pass

        if '로' in word:
            # '로' 기준
            sub_list = word.split('로') 
            word_list[i:i+1] = sub_list
        else:
            pass

        if '의' in word:
            # '의' 기준
            sub_list = word.split('의') 
            word_list[i:i+1] = sub_list
        else:
            pass
        
        if '와' in word:
            # '와' 기준
            sub_list = word.split('와') 
            word_list[i:i+1] = sub_list
        else:
            pass

        if '으로' in word:
            # '으로' 기준
            sub_list = word.split('으로') 
            word_list[i:i+1] = sub_list
        else:
            pass

    # 출력 파일 열기 (덮어쓰기 모드)
    with open(text_file, 'w', encoding='UTF-8') as file:
        # 리스트 요소를 문자열로 결합하여 파일에 쓰기
        file.write(' '.join(word_list))
        
    ##########################################################################
    
    # 데이터베이스 연동 코드(2023.06.03)
    conn = mysql.connect()
    cursor = conn.cursor()
    
    sql = "SELECT test_list FROM testinterview ORDER BY RAND() LIMIT 1"
    cursor.execute(sql)
    
    data = cursor.fetchone()
    
    cursor.close()
    conn.close()
    
    if data:
        test_list = data[0]
        return render_template('test2.html', text=text, runningTime=round(runningTime), wordCountResult=wordCountResult, wordSpeed=wordSpeed, test_list=test_list)
    else:
        return "데이터를 가져오지 못했습니다."


if __name__ == '__main__':
    app.run(debug=True, threaded=True)
   