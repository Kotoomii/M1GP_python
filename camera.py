import cv2
import numpy as np
from screeninfo import get_monitors
import os

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

def overlay_smile(face_box, frame, smile_img):
    """
    顔部分にスマイル画像を合成

    Args:
        face_box (array): 顔の位置を示すボックス情報 (x, y, w, h)
        frame (array): カメラからのフレーム
        smile_img (array): スマイル画像

    Returns:
        frame (array): スマイル画像を合成したフレーム
    """
    x, y, w, h = face_box.astype(int)
    
    # スマイル画像を顔のサイズにリサイズ
    smile_resized = cv2.resize(smile_img, (w, h))
    
    # スマイル画像を顔の位置に合成
    if smile_resized.shape[2] == 4:  # スマイル画像がRGBAの場合
        alpha_s = smile_resized[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        
        for c in range(0, 3):
            frame[y:y+h, x:x+w, c] = (alpha_s * smile_resized[:, :, c] +
                                      alpha_l * frame[y:y+h, x:x+w, c])
    else:
        frame[y:y+h, x:x+w] = smile_resized
    
    return frame

def main():
    cap = initialize_camera()
    if not cap:
        return

    model = load_model()
    if not model:
        return

    # スマイル画像の読み込み
    smile_img_path = 'face/smile.png'
    if not os.path.exists(smile_img_path):
        print(f"スマイル画像が見つかりません: {smile_img_path}")
        return
    smile_img = cv2.imread(smile_img_path, cv2.IMREAD_UNCHANGED)

    display_mode = 1  # 0: 通常のカメラ映像, 1: スマイル画像を合成

    # モニターの解像度を取得
    monitor = get_monitors()[0]
    screen_width = monitor.width
    screen_height = monitor.height

    while True:
        # カメラのフレームを常に表示
        ret, frame = cap.read()
        if ret:
            height, width, channels = frame.shape

            if display_mode == 1:
                # 顔検出
                model.setInputSize((width, height))
                faces = model.detect(frame)
                if faces[1] is not None:
                    for face in faces[1]:
                        box = face[:4]
                        frame = overlay_smile(box, frame, smile_img)

            # フルスクリーンのサイズに合わせてカメラのフレームをリサイズ
            frame = cv2.resize(frame, (screen_width, screen_height))

            # ミラー表示
            cv2.imshow("Camera", frame)

            # デバッグ用
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                display_mode = 1 - display_mode  # display_modeを切り替え

        else:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
