from google.cloud import vision
import speech_recognition as sr
import cv2
import socket
import random

def take_photo():
    """
    カメラで写真を撮影し、ファイルに保存する。

    Returns:
        str: 保存された写真のファイルパス。失敗した場合はNone。
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("カメラが起動できません")
        return None

    ret, frame = cap.read()
    if ret:
        cv2.imwrite('image/photo.jpg', frame)
        print("写真を保存しました。")
        cap.release()
        return 'image/photo.jpg'
    cap.release()
    return None

def determine_emotion(emotions):
    """
    渡された感情スコアから最も強い感情を決定する。

    Args:
        emotions (dict): 感情スコアの辞書。

    Returns:
        str: 最も強い感情の名前。
    """
    # 全ての値が2以下、または全てが同じ値の場合は"magao"を返す
    if all(value == 2 or value == 1 for value in emotions.values()) or len(set(emotions.values())) == 1:
        return "magao"

    # 最大値を持つ感情を探す
    max_value = max(emotions.values())
    max_emotions = [emotion for emotion, value in emotions.items() if value == max_value]

    # 最大値を持つ感情が複数ある場合はランダムに選ぶ
    if len(max_emotions) > 1:
        return random.choice(max_emotions)

    return max_emotions[0]

def detect_faces(path):
    """
    画像内の顔を検出し、感情を数値に変換する。

    Args:
        path (str): 画像ファイルのパス。

    Returns:
        list: 顔検出結果と感情スコアのリスト。
    """
    client = vision.ImageAnnotatorClient()
    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.face_detection(image=image)
    faces = response.face_annotations
    likelihood_name = {
        "UNKNOWN": 0,
        "VERY_UNLIKELY": 1,
        "UNLIKELY": 2,
        "POSSIBLE": 3,
        "LIKELY": 4,
        "VERY_LIKELY": 5
    }

    face_results = []
    for face in faces:
        emotions = {
            "anger": likelihood_name[vision.Likelihood(face.anger_likelihood).name],
            "joy": likelihood_name[vision.Likelihood(face.joy_likelihood).name],
            "surprise": likelihood_name[vision.Likelihood(face.surprise_likelihood).name],
            "sorrow": likelihood_name[vision.Likelihood(face.sorrow_likelihood).name]
        }
        dominant_emotion = determine_emotion(emotions)
        face_bounds = [(vertex.x, vertex.y) for vertex in face.bounding_poly.vertices]
        face_results.append({
            "emotions": emotions,
            "dominant_emotion": dominant_emotion,
            "bounds": face_bounds
        })
    if response.error.message:
        raise Exception("{}\nFor more info on error messages, check: https://cloud.google.com/apis/design/errors".format(response.error.message))

    print("Faces:")
    for result in face_results:
        print(f"Emotions: {result['emotions']}")
        print(f"Dominant Emotion: {result['dominant_emotion']}")
        print("Face bounds: {}".format(",".join(f"({x},{y})" for x, y in result['bounds'])))
        print('----')
    
    return face_results

def emotion_to_code(dominant_emotion):
    """
    感情名を対応するコードに変換する。

    Args:
        dominant_emotion (str): 感情の名前。

    Returns:
        int: 感情に対応するコード。
    """
    emotion_code = {
        "anger": 1,
        "joy": 2,
        "surprise": 3,
        "sorrow": 4,
        "magao": 5
    }
    return emotion_code.get(dominant_emotion, 5)  # デフォルトは5 (magao)

def main():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    # ソケットの設定
    host = 'localhost'
    port = 12345
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))

    while True:
        with mic as source:
            print("何かお話ししてください。")
            audio = recognizer.listen(source)

        try:
            # 音声をテキストに変換
            text = recognizer.recognize_google(audio, language="ja-JP")
            print(f"Recognized: {text}")
            if text == "こんにちは":
                photo_path = take_photo()
                if photo_path:
                    # 顔検出と感情分析
                    face_results = detect_faces(photo_path)
                    if face_results:
                        dominant_emotion = face_results[0]['dominant_emotion']
                        emotion_code = emotion_to_code(dominant_emotion)
                        # ソケットに感情コードを送信
                        client_socket.sendall(str(emotion_code).encode())
                        print(f"display_modeを{emotion_code}に設定しました。")
            elif text == "終わりだよ":
                client_socket.sendall(b'0')
                print("終了します。")
                break
        except sr.UnknownValueError:
            print("Google Web Speech APIは音声を認識できませんでした。")
        except sr.RequestError as e:
            print(f"Google Web Speech APIに音声認識を要求できませんでした; {e}")

    client_socket.close()

if __name__ == "__main__":
    main()
