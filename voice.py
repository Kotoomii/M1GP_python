from google.cloud import vision
import speech_recognition as sr
import cv2
import socket
import random

def take_photo():
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
    """Determine the dominant emotion from the given emotion scores."""
    if all(value == 2 or value == 1 for value in emotions.values()) or len(set(emotions.values())) == 1:
        return "magao"

    max_value = max(emotions.values())
    max_emotions = [emotion for emotion, value in emotions.items() if value == max_value]

    if len(max_emotions) > 1:
        return random.choice(max_emotions)

    return max_emotions[0]

def detect_faces(path):
    """Detects faces in an image and converts emotions to numerical values."""
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
    """Convert dominant emotion to corresponding code."""
    emotion_code = {
        "anger": 1,
        "joy": 2,
        "surprise": 3,
        "sorrow": 4,
        "magao": 5
    }
    return emotion_code.get(dominant_emotion, 5)  # Default to 5 (magao) if not found

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
            text = recognizer.recognize_google(audio, language="ja-JP")
            print(f"Recognized: {text}")
            if text == "こんにちは":
                photo_path = take_photo()
                if photo_path:
                    face_results = detect_faces(photo_path)
                    if face_results:
                        dominant_emotion = face_results[0]['dominant_emotion']
                        emotion_code = emotion_to_code(dominant_emotion)
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
