# Demo ADAS local: YOLOv8 + YOLOPv2

Nhận diện người, xe, đèn giao thông (đỏ/vàng/xanh), mặt đường chạy được, vạch kẻ làn
và suy ra làn đang chạy. Dùng model có sẵn, kèm fine-tune model phân loại màu đèn. Giao diện Gradio.

## Cài đặt (Windows, PowerShell)

```
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Kiểm tra GPU, phải in ra `True` và tên card:

```
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Weights YOLOPv2 tải thủ công tại https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt
Weights YOLOv8s tải thủ công tại https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt
rồi bỏ vào thư mục `weights/`.

## Chạy

```
python app.py
```

Mở http://127.0.0.1:7860.

## Samples

Thư mục `samples/` có sẵn các mẫu photos, videos, lấy từ các repo GitHub giấy phép
MIT/Apache-2.0. Nguồn từng video trong `videos/NGUON_VIDEO.txt`.

## Cấu trúc

| File | Vai trò |
|---|---|
| `adas/objects.py` | YOLOv8 COCO: người, xe, đèn; phân loại màu đèn bằng HSV |
| `adas/road.py` | YOLOPv2: mặt nạ mặt đường chạy được + vạch kẻ làn |
| `adas/ego_lane.py` | Heuristic tách làn đang chạy khỏi mặt đường |
| `adas/visualize.py` | Vẽ overlay, bbox, dải cảnh báo |
| `adas/pipeline.py` | Ghép các bước, sinh cảnh báo |
| `app.py` | Giao diện Gradio (ảnh + video) |

## Giới hạn đã biết

- Làn đang chạy là suy luận heuristic, giả định camera đặt giữa xe; dễ sai khi
  vạch mờ, đường không kẻ vạch, xe đang chuyển làn.
- Màu đèn dùng HSV nên dễ sai với đèn quá nhỏ/xa, ban đêm, đèn đếm ngược.
- YOLOPv2 train trên đường Mỹ (BDD100K), đường Việt Nam nhiều xe máy có thể kém hơn.
- Ảnh không phải tỉ lệ 16:9 bị co giãn nhẹ khi đưa vào YOLOPv2.
- Vạch qua đường (vạch zebra) bị YOLOPv2 nhận nhầm là vạch làn, làm vùng làn đang chạy bị vỡ.
