import time
import numpy as np
import cv2
import subprocess  # 音を鳴らすために追加
from picamera2 import Picamera2
from tflite_runtime.interpreter import Interpreter


MODEL_PATH = "/home/omoto/edge-ai/model_quant.tflite"
BOIL_CLASS_INDEX = 0
BOIL_THRESHOLD = 0.1
BOIL_STABLE_FRAMES = 5
CAMERA_SIZE = (640, 480)


# --- 音の設定（一番音が大きい設定を反映） ---
AUDIO_DEVICE = "plughw:0,0"
FREQ = "440"
RATE = "48000"

def play_boil_alarm():
    """沸騰を知らせる大音量アラーム（0.8秒間）"""
    # ユーザー様が成功したコマンドをそのまま活用し、timeoutで長さを制御
    cmd = [
        "timeout", "0.8s", 
        "speaker-test", "-D", AUDIO_DEVICE, "-c2", "-t", "sine", "-f", FREQ, "-r", RATE
    ]
    # カメラ処理に戻るため、実行してすぐ終了するようにします
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)




def load_model(model_path):
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details


def preprocess(frame_rgb, input_details):
    #前処理
    _, input_h, input_w, _ = input_details[0]["shape"]
    h, w, _ = frame_rgb.shape
    # 一応チェック（デバッグ用）
    if h < input_h + 5 or w < input_w + 5:
        raise ValueError(
            f"Frame too small for crop: frame=({w}x{h}), "
            f"need at least ({input_w+5}x{input_h+5})"
        )
    top = 5
    left = 5
    crop = frame_rgb[top:top+input_h, left:left+input_w, :]

    input_data = crop.astype(np.float32) / 255.0

    # 量子化情報を取得
    scale, zero_point = input_details[0]["quantization"]
    input_type = input_details[0]["dtype"]

    if input_type == np.uint8 and scale > 0:
        # float(0〜1) → 量子化 uint8
        input_data = input_data / scale + zero_point
        input_data = np.clip(input_data, 0, 255).astype(np.uint8)
    else:
        # 非量子化モデルなどの場合
        input_data = input_data.astype(input_type)

    # バッチ次元を追加
    input_data = np.expand_dims(input_data, axis=0)
    return input_data




def postprocess(output_data, output_details):
    scale, zero_point = output_details[0]["quantization"]
    if scale > 0:
        # 量子化された uint8 -> float に戻す
        output_float = (output_data.astype(np.float32) - zero_point) * scale
    else:
        output_float = output_data.astype(np.float32)

    #一応0：１でクリップ
    raw = float(output_float.reshape(-1)[0])
    raw = max(0.0, min(1.0,raw))


    boil_prob = 1.0 - raw
    return boil_prob


def main():
    interpreter, input_details, output_details = load_model(MODEL_PATH)
    input_index = input_details[0]["index"]
    output_index = output_details[0]["index"]

    picam2 = Picamera2()
    camera_config = picam2.create_preview_configuration(
        main={"size": CAMERA_SIZE, "format": "RGB888"}
    )
    picam2.configure(camera_config)
    picam2.start()
    time.sleep(2.0)  # 露光が落ち着くまで少し待つ

    print("Boiling detection started. Press Ctrl+C to stop.")

    boil_count = 0

    try:
      while True:
            frame_rgb = picam2.capture_array()  # RGB888

            # 前処理
            input_data = preprocess(frame_rgb, input_details)

            # 推論
            interpreter.set_tensor(input_index, input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_index)
            boil_prob = postprocess(output_data, output_details)

            # 沸騰判定（単純なしきい値 + 連続フレーム）
            if boil_prob >= BOIL_THRESHOLD:
                boil_count += 1
            else:
                boil_count = max(0, boil_count - 1)
            is_boiling = boil_count >= BOIL_STABLE_FRAMES
            if is_boiling:
                print("【警告】沸騰しています！火を止めてください！")
                play_boil_alarm() # ここで音が鳴ります



            # 状態表示（今はターミナルに表示）
            print(
                f"Boil score: {boil_prob:.3f}, "
                f"count: {boil_count}, "
                f"BOILING: {is_boiling}"
            )


            time.sleep(0.5)  # ループ間隔（20fpsくらい）

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        picam2.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
