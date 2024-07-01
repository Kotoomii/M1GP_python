from google.cloud import vision
import speech_recognition as sr
import cv2
import socket

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

def detect_faces(path):
    """Detects faces in an image."""
    client = vision.ImageAnnotatorClient()
    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.face_detection(image=image)
    faces = response.face_annotations
    likelihood_name = ("UNKNOWN", "VERY_UNLIKELY", "UNLIKELY", "POSSIBLE", "LIKELY", "VERY_LIKELY")

    print("Faces:")
    print(faces)
    print('----')
    for face in faces:
        print(f"anger: {likelihood_name[face.anger_likelihood]}")
        print(f"joy: {likelihood_name[face.joy_likelihood]}")
        print(f"surprise: {likelihood_name[face.surprise_likelihood]}")
        vertices = [f"({vertex.x},{vertex.y})" for vertex in face.bounding_poly.vertices]
        print("face bounds: {}".format(",".join(vertices)))

    if response.error.message:
        raise Exception("{}\nFor more info on error messages, check: https://cloud.google.com/apis/design/errors".format(response.error.message))

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
                client_socket.sendall(b'1')
                print("display_modeを1に設定しました。")
                photo_path = take_photo()
                if photo_path:
                    detect_faces(photo_path)
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
