import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def analyze_healthy_patterns():
    # Đọc dữ liệu
    data = pd.read_csv('Data.csv')
    
    # Tách dữ liệu thành nhóm khỏe mạnh và không khỏe mạnh
    healthy_data = data[data['target'] == 1]
    unhealthy_data = data[data['target'] == 0]
    
    # Phân tích từng đặc trưng
    print("\nPHÂN TÍCH MẪU KHỎE MẠNH:")
    print("=========================")
    
    # 1. Phân tích các đặc trưng categorical
    categorical_features = ['Age', 'trest', 'chol', 'thalach', 'oldpeak']
    
    print("\n1. Phân tích đặc trưng categorical:")
    print("-----------------------------------")
    for feature in categorical_features:
        print(f"\n{feature}:")
        value_counts = healthy_data[feature].value_counts()
        total = len(healthy_data)
        for value, count in value_counts.items():
            percentage = (count / total) * 100
            print(f"- {value}: {percentage:.1f}% ({count} cases)")
            
        # Vẽ biểu đồ so sánh
        plt.figure(figsize=(10, 6))
        pd.crosstab(data[feature], data['target'], normalize='columns').plot(kind='bar')
        plt.title(f'Phân bố {feature} theo tình trạng sức khỏe')
        plt.xlabel(feature)
        plt.ylabel('Tỷ lệ')
        plt.legend(['Không khỏe mạnh', 'Khỏe mạnh'])
        plt.tight_layout()
        plt.savefig(f'analysis_{feature}.png')
        plt.close()
    
    # 2. Phân tích các đặc trưng số
    numeric_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    
    print("\n2. Phân tích đặc trưng số:")
    print("---------------------------")
    for feature in numeric_features:
        print(f"\n{feature}:")
        value_counts = healthy_data[feature].value_counts()
        total = len(healthy_data)
        for value, count in value_counts.items():
            percentage = (count / total) * 100
            print(f"- Giá trị {value}: {percentage:.1f}% ({count} cases)")
    
    # 3. Đưa ra gợi ý các giá trị tốt nhất
    print("\nGỢI Ý GIÁ TRỊ TỐT NHẤT CHO SỨC KHỎE TIM MẠCH:")
    print("=============================================")
    
    suggestions = {
        'Age': healthy_data['Age'].mode()[0],
        'sex': healthy_data['sex'].mode()[0],
        'cp': healthy_data['cp'].mode()[0],
        'trest': healthy_data['trest'].mode()[0],
        'chol': healthy_data['chol'].mode()[0],
        'fbs': healthy_data['fbs'].mode()[0],
        'restecg': healthy_data['restecg'].mode()[0],
        'thalach': healthy_data['thalach'].mode()[0],
        'exang': healthy_data['exang'].mode()[0],
        'oldpeak': healthy_data['oldpeak'].mode()[0],
        'slope': healthy_data['slope'].mode()[0],
        'ca': healthy_data['ca'].mode()[0],
        'thal': healthy_data['thal'].mode()[0]
    }
    
    descriptions = {
        'Age': 'Độ tuổi',
        'sex': 'Giới tính (0: Nữ, 1: Nam)',
        'cp': 'Loại đau ngực (0-3)',
        'trest': 'Huyết áp lúc nghỉ',
        'chol': 'Mức cholesterol',
        'fbs': 'Đường huyết lúc đói > 120 mg/dl',
        'restecg': 'Kết quả điện tâm đồ lúc nghỉ (0-2)',
        'thalach': 'Nhịp tim tối đa',
        'exang': 'Đau thắt ngực khi tập thể dục',
        'oldpeak': 'ST depression',
        'slope': 'Độ dốc của đoạn ST peak',
        'ca': 'Số lượng mạch máu chính',
        'thal': 'Thalassemia'
    }
    
    print("\nGiá trị phổ biến nhất trong nhóm khỏe mạnh:")
    for feature, value in suggestions.items():
        healthy_count = len(healthy_data[healthy_data[feature] == value])
        percentage = (healthy_count / len(healthy_data)) * 100
        print(f"\n{descriptions[feature]}:")
        print(f"- Giá trị tốt nhất: {value}")
        print(f"- Tỷ lệ trong nhóm khỏe mạnh: {percentage:.1f}%")
    
    # 4. Tạo bảng tổng hợp
    print("\nBẢNG TỔNG HỢP KHUYẾN NGHỊ:")
    print("==========================")
    print("\nĐể có sức khỏe tim mạch tốt, nên duy trì các chỉ số sau:")
    
    recommendations = {
        'Huyết áp': 'Normal',
        'Cholesterol': 'Normal',
        'Đường huyết': 'Bình thường (< 120 mg/dl)',
        'Nhịp tim': 'Normal đến High',
        'Đau ngực': 'Không có hoặc ít',
        'Tập thể dục': 'Không đau thắt ngực khi tập',
        'ST depression': 'Low đến Normal',
        'Số lượng mạch máu': 'Càng ít càng tốt (0-1)'
    }
    
    for aspect, recommendation in recommendations.items():
        print(f"\n{aspect}:")
        print(f"- Khuyến nghị: {recommendation}")

if __name__ == "__main__":
    analyze_healthy_patterns() 