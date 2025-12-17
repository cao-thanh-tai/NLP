# NLP

## Cài đặt môi trường

### Khởi tạo môi trường
```bash
conda env create -f environment.yml
```

### Kích hoạt môi trường
```bash
conda activate nlp
```

## Sử dụng

### Huấn luyện mô hình
Huấn luyện mô hình Seq2Seq với tập dữ liệu Multi30k (EN→DE):
```bash
python .\src\train.py
```

### Dịch câu tương tác
Chạy chương trình dịch câu từ tiếng Anh sang tiếng Đức (interactive mode):
```bash
python .\src\main.py
```

### Dịch câu đơn lẻ
Sử dụng module translate để dịch câu:
```bash
python .\src\translate.py
```

