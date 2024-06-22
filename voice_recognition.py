import cv2
from google.cloud import vision
import speech_recognition as sr

def take_photo():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("カメラが起動できません")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 幅を1280に設定
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # 高さを720に設定
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)  # 明るさを調整 (0から1の範囲で設定)
    cap.set(cv2.CAP_PROP_CONTRAST, 0.8)  # コントラストを調整 (通常0から1の範囲)

    ret, frame = cap.read()
    if ret:
        cv2.imwrite('image/photo.jpg', frame)
        print("写真を保存しました。")
    
    cap.release()
    cv2.destroyAllWindows()
    
    return 'image/photo.jpg'

def detect_faces(path):
    """Detects faces in an image."""
    client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.face_detection(image=image)
    faces = response.face_annotations

    likelihood_name = (
        "UNKNOWN", "VERY_UNLIKELY", "UNLIKELY",
        "POSSIBLE", "LIKELY", "VERY_LIKELY",
    )
    print("Faces:")
    for face in faces:
        print(f"anger: {likelihood_name[face.anger_likelihood]}")
        print(f"joy: {likelihood_name[face.joy_likelihood]}")
        print(f"surprise: {likelihood_name[face.surprise_likelihood]}")
        vertices = [f"({vertex.x},{vertex.y})" for vertex in face.bounding_poly.vertices]
        print("face bounds: {}".format(",".join(vertices)))

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )

def main():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

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
                    detect_faces(photo_path)
            elif text == "終わりだよ":
                print("終了します。")
                break
        except sr.UnknownValueError:
            print("Google Web Speech APIは音声を認識できませんでした。")
        except sr.RequestError as e:
            print(f"Google Web Speech APIに音声認識を要求できませんでした; {e}")

if __name__ == "__main__":
    main()
