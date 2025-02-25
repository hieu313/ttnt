# Giải thích chi tiết code trong file main.py

## Tổng quan

File `main.py` chứa một hệ thống chẩn đoán bệnh tim sử dụng cây quyết định (Decision Tree). Hệ thống này đọc dữ liệu từ file CSV, xử lý dữ liệu, xây dựng cây quyết định và tạo ra các luật để chẩn đoán.

## Import các thư viện cần thiết

```python
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
```

- `sys`: Thư viện hệ thống
- `numpy`: Thư viện xử lý số liệu
- `pandas`: Thư viện xử lý dữ liệu dạng bảng
- `LabelEncoder`: Công cụ chuyển đổi dữ liệu categorical sang dạng số

## Class HealthPredictor

### Constructor

```python
def __init__(self):
    self.label_encoders = {}
```

- Khởi tạo dictionary để lưu trữ các bộ encoder cho từng thuộc tính. (các thuộc tính có value là chữ)
- Để tính toán thì các value chữ (Adult, Old...) phải chuyển sang dạng số
- Khi đó các cột chứa value chữ sẽ được lưu trong dict trên (nếu trong Data.csv: thí nó chứa các cột Age, trest, chol, thalach, oldpeak)

### Các phương thức xử lý dữ liệu

#### load_data

```python
def load_data(self, path="Data.csv"):
  data = pd.read_csv(path) # đọc dữ liệu từ file
    columns_to_process = [col for col in data.columns if col != 'target'] # loại bỏ cột target
    for col in columns_to_process:
        # Kiểm tra xem cột có phải có giá trị toàn số nguyên không
        if not data[col].dtype.kind in 'iu':  # 'i' cho integer, 'u' cho unsigned integer
          print(f"Xử lý cột {col}")
          self.label_encoders[col] = LabelEncoder()
          data[col] = self.label_encoders[col].fit_transform(data[col])
    return data
```

- Đọc file CSV
- Chuyển đổi các cột có dữ liệu không phải số nguyên (ví dụ: cột Age) sang dạng số sử dụng LabelEncoder
- Lưu các encoder để sử dụng sau này

#### convert_value_to_label

```python
def convert_value_to_label(self, attribute, value):
    if isinstance(value, str):
        return value

    # Nếu thuộc tính có trong label_encoders thì chuyển đổi
    if attribute in self.label_encoders:
        try:
            return self.label_encoders[attribute].inverse_transform([value])[0]
        except ValueError:
            # Nếu không thể chuyển đổi, trả về giá trị gốc
            return value
    return value
```

Chuyển đổi giá trị số thành nhãn tương ứng cho thuộc tính.

- Input: tên thuộc tính và giá trị số
- Output: nhãn tương ứng

- Ví dụ: `convert_value_to_label('Age', 2)` thì sẽ trả về 'Old'
- Có thể xem [ở đây](./result/label_encoders.json)

#### load_full_attributes

```python
def load_full_attributes(data):
```

Lấy danh sách tất cả các thuộc tính (trừ cột target).

#### save_processed_data và save_label_encoders

```python
def save_processed_data(self, data, output_path="processed_data.csv")
def save_label_encoders(self, output_path="lable_encoders.json")
```

Lưu dữ liệu đã xử lý (chuyển từ dạng chữ sang dạng số) và thông tin mapping của các encoder ra file.

### Các phương thức tính toán Entropy và Information Gain

#### cal_entropy

```python
@staticmethod
def cal_entropy(data):
```

Tính entropy của tập dữ liệu:

- Đếm số lượng mỗi nhãn
- Tính xác suất của mỗi nhãn
- Tính entropy theo công thức: -Σ(p \* log2(p))

#### partition

```python
@staticmethod
def partition(data, attribute):
```

Phân chia dữ liệu thành các nhóm theo giá trị của thuộc tính.

#### calculate_information_gain

```python
def calculate_information_gain(self, data, attribute):
```

Tính Information Gain cho một thuộc tính:

- Tính entropy hiện tại
- Tính entropy sau khi phân chia theo thuộc tính
- Information Gain = entropy hiện tại - entropy sau khi phân chia

### Các phương thức xây dựng cây quyết định

#### find_best_attribute

```python
def find_best_attribute(self, data, attributes):
```

Tìm thuộc tính tốt nhất để phân chia dữ liệu dựa trên Information Gain.

#### build_decision_tree

- Xây dựng cây quyết định với giới hạn độ sâu

```python
def build_decision_tree(self, data, attributes, max_depth=5, current_depth=0):
  # Xây dựng cây quyết định với giới hạn độ sâu
  if (current_depth >= max_depth or  # đạt độ sâu tối đa
          len(attributes) == 0 or  # hết thuộc tính
          len(data) == 0 or  # hết dữ liệu
          len(data['target'].unique()) == 1):  # chỉ còn 1 nhãn
      return data['target'].mode()[0] # trả về giá trị target phổ biến nhất trong tập dữ liệu hiện tại
  best_attr = self.find_best_attribute(data, attributes)
  sub_tree = {}

  # Loại bỏ thuộc tính đã sử dụng khỏi danh sách
  new_attributes = [f for f in attributes if f != best_attr]

  # Xây dựng các nhánh con
  for value in sorted(data[best_attr].unique()):
      # Tạo tập dữ liệu con chứa các mẫu có giá trị thuộc tính tương ứng
      sub_data = data[data[best_attr] == value].copy()
      # tập dữ liệu con không rỗng:
      if len(sub_data) > 0:
          # Chuyển đổi giá trị số thành nhãn có ý nghĩa (từ 1 thành Adult)
          label_value = self.convert_value_to_label(best_attr, value)
          # đệ quy: tạo dữ liệu con, danh sách thuộc tính mới, tăng độ sâu lên 1
          sub_tree[label_value] = self.build_decision_tree(
              sub_data,
              new_attributes,
              max_depth,
              current_depth + 1
          )
      # tập dữ liệu con rỗng:
      else:
          # Sử dụng value target phổ biến nhất của tập dữ liệu cha
          label_value = self.convert_value_to_label(best_attr, value)
          sub_tree[label_value] = data['target'].mode()[0]

  return {best_attr: sub_tree}
```

- Dữ liệu đầu vào

  - `data`: DataFrame chứa dữ liệu huấn luyện
  - `attributes`: Danh sách các thuộc tính để xây dựng cây
  - `max_depth`: Độ sâu tối đa của cây (mặc định là 5)
  - `current_depth`: Độ sâu hiện tại của nút trong cây (bắt đầu từ 0)

### Các phương thức tạo và quản lý luật

#### generate_rules

```python
def generate_rules(self, tree, attributes,  rule_values=None):
    if rule_values is None:
        # Khởi tạo dictionary với tất cả thuộc tính là null
        rule_values = {attr: None for attr in attributes}

    rules = []

    # Nếu node là lá (giá trị dự đoán)
    if not isinstance(tree, dict):
        return [(rule_values.copy(), "Bệnh tim" if tree == 1 else "Không bệnh tim")]

    # Lấy thuộc tính gốc và các nhánh
    attribute = list(tree.keys())[0]
    branches = tree[attribute]

    # Duyệt qua từng nhánh
    for value, subtree in branches.items():
        # Cập nhật giá trị cho thuộc tính hiện tại
        new_rule_values = rule_values.copy()
        new_rule_values[attribute] = value

        # Đệ quy với cây con
        rules.extend(self.generate_rules(subtree, attributes, new_rule_values))

    return rules
```

Tạo tập luật từ cây quyết định bằng cách duyệt qua các nhánh.

#### print_rules và save_rules_to_csv

```python
def print_rules(rules_df, filename='result/rules.txt')
def save_rules_to_csv(self, rules, filename='rules.csv')
```

In và lưu tập luật ra file.

### Phương thức chẩn đoán

#### diagnose

```python
def diagnose(self, patient_data, rules_df):
```

Chẩn đoán bệnh nhân dựa trên tập luật:

- Chuyển đổi dữ liệu bệnh nhân sang dạng nhãn
- So sánh với từng luật
- Trả về kết quả chẩn đoán và luật phù hợp

### Hàm main và hàm phụ trợ

#### create_test_patient

```python
def create_test_patient():
```

Tạo dữ liệu test cho một bệnh nhân.

#### main

```python
def main():
```

Hàm chính để chạy chương trình:

- Khởi tạo HealthPredictor
- Chạy chẩn đoán với dữ liệu test
