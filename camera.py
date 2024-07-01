import cv2
import numpy as np
from screeninfo import get_monitors
import os
import socket
import threading

def initialize_camera():
    """
    カメラ初期化、全画面表示ウィンドウ作成

    Returns:
        cap (cv2.VideoCapture): カメラキャプチャオブジェクト. カメラの初期化に失敗した場合はNone
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("カメラが起動できません")
        return None
    
    # ウィンドウを作成し、全画面表示を設定
    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    return cap

def load_model():
    """
    顔検出モデルのロード

    Returns:
        model (cv2.FaceDetectorYN): 顔検出モデルオブジェクト. モデルの読み込みに失敗した場合はNone
    """

    # 顔検出用モデルファイル
    model_path = "./model/face_detection_yunet_2023mar.onnx"

    if not os.path.exists(model_path):
        print(f"モデルファイルが見つかりません: {model_path}")
        return None

    try:
        model = cv2.FaceDetectorYN.create(
            model_path,
            "",
            (320, 320),
            0.9,
            0.3,
            5000
        )
    except cv2.error as e:
        print(f"モデルの読み込みに失敗しました: {e}")
        return None

    return model

def load_emotion_images():
    """
    感情画像をロード

    Returns:
        emotion_images (dict): 感情画像を格納した辞書
    """
    emotions = ["anger", "joy", "surprise", "sorrow", "magao"]
    emotion_images = {}

    for emotion in emotions:
        img_path = f'face/{emotion}.png'
        if os.path.exists(img_path):
            emotion_images[emotion] = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        else:
            print(f"画像が見つかりません: {img_path}")

    return emotion_images

def overlay_emotion(face_box, frame, emotion_img):
    """
    顔部分に感情画像を合成

    Args:
        face_box (array): 顔の位置を示すボックス情報 (x, y, w, h)
        frame (array): カメラからのフレーム
        emotion_img (array): 感情画像

    Returns:
        frame (array): 感情画像を合成したフレーム
    """
    x, y, w, h = face_box.astype(int)
    
    # 感情画像を顔のサイズにリサイズ
    emotion_resized = cv2.resize(emotion_img, (w, h))
    
    # 感情画像を顔の位置に合成
    if emotion_resized.shape[2] == 4:  # 感情画像がRGBAの場合
        alpha_s = emotion_resized[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        
        for c in range(0, 3):
            frame[y:y+h, x:x+w, c] = (alpha_s * emotion_resized[:, :, c] +
                                      alpha_l * frame[y:y+h, x:x+w, c])
    else:
        frame[y:y+h, x:x+w] = emotion_resized
    
    return frame

def socket_server():
    global display_mode
    host = 'localhost'
    port = 12345
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    print("ソケットサーバーが起動しました。")

    conn, addr = server_socket.accept()
    print(f"接続されました: {addr}")

    while True:
        data = conn.recv(1024)
        if not data:
            break
        display_mode = int(data.decode())
        print(f"display_modeが{display_mode}に設定されました。")

    conn.close()
    server_socket.close()

def main():
    global display_mode
    display_mode = 0  # 初期値を0に設定

    # ソケットサーバーを別スレッドで起動
    server_thread = threading.Thread(target=socket_server)
    server_thread.start()

    cap = initialize_camera()
    if not cap:
        return

    model = load_model()
    if not model:
        return

    # 感情画像の読み込み
    emotion_images = load_emotion_images()

    # モニターの解像度を取得
    monitor = get_monitors()[0]
    screen_width = monitor.width
    screen_height = monitor.height

    emotion_map = {
        1: "anger",
        2: "joy",
        3: "surprise",
        4: "sorrow",
        5: "magao"
    }

    while True:
        # カメラのフレームを常に表示
        ret, frame = cap.read()
        if ret:
            height, width, channels = frame.shape

            if display_mode in emotion_map:
                emotion = emotion_map[display_mode]
                if emotion in emotion_images:
                    emotion_img = emotion_images[emotion]

                    # 顔検出
                    model.setInputSize((width, height))
                    faces = model.detect(frame)
                    if faces[1] is not None:
                        for face in faces[1]:
                            box = face[:4]
                            frame = overlay_emotion(box, frame, emotion_img)

            # フルスクリーンのサイズに合わせてカメラのフレームをリサイズ
            frame = cv2.resize(frame, (screen_width, screen_height))

            # ミラー表示
            cv2.imshow("Camera", frame)

            # デバッグ用
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        else:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
