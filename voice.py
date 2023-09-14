import speech_recognition as sr
import os
import time
import sys
sys.stdout.reconfigure(encoding='utf-8')


# 녹음을 시작하는 함수 : startRecord()
def startRecord(text_file):
    result = {}
    # txt 파일 생성
    with open(text_file, 'w', encoding='UTF-8') as file:
        pass

    r = sr.Recognizer()

    # 마이크 사용
    with sr.Microphone() as source:
        # 녹음이 시작됨과 동시에 시간 측정
        start_time = time.time()    
        audio = r.listen(source)
        text = r.recognize_google(audio, language='ko-KR') # 음성인식한 텍스트

        # print("사용자의 답변 : " + text)
        result["text"] = text # 추가(2023.05.08)

        #텍스트 파일에 결과 추가
        with open(text_file, 'a', encoding='UTF-8') as file:
            file.write(text + "\n")

        # 녹음 종료 시간 측정
        end_time = time.time()
        
        # 실행시간 = 종료시 - 시작시
        runningTime = end_time - start_time
        
        result["runningTime"] = runningTime # 추가(2023.05.08)

        return result # 추가(2023.05.08)
    

# 글자 수를 계산해야지 말하기 속도 구하기 가능 (Line 62 ~ Line 77)
# startRecord() 반환값 이용
def fileopen(data):
    with open(data, 'r', encoding='UTF-8') as file:
        text = file.read()
        splitdata = text.split()
 
    return splitdata, len(splitdata) # word_len
 
 
def count_character(data):
    count = 0  
    for i in data :  
        count += len(i)
 
    return  count


# 말하기 빠르기 비교 함수 (Line 84 ~ Line 97)
def compareSpeakingVel(wordCountResult, final_runningTime, result): # 매개변수 추가(2023.05.08)
    calResult_min = 0.2 * wordCountResult 
    calResult_max = 0.3 * wordCountResult 

    #비교 시작
    #startRecord 함수를 보면, runningTime을 반환하게 만들어둠
    if calResult_min > final_runningTime:
        result["speed"] = "말하는 속도가 너무 빠릅니다"
    elif calResult_max < final_runningTime:
        result["speed"] = "말하는 속도가 너무 느립니다"
    else:
        result["speed"] = "적당한 속도로 말헀습니다"
    
    return result["speed"]