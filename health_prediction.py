import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime
import warnings

# Tắt các cảnh báo không cần thiết
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

class HeartDiseasePredictor:
    def __init__(self):
        # Mô tả ý nghĩa của các đặc trưng
        self.feature_descriptions = {
            'Age': 'Độ tuổi (Adult/MidleAge/Old)',
            'sex': 'Giới tính (0: Nữ, 1: Nam)',
            'cp': 'Loại đau ngực (0-3)',
            'trest': 'Huyết áp lúc nghỉ (High/Normal/Low)',
            'chol': 'Mức cholesterol (Normal/High Risk/Extreme)',
            'fbs': 'Đường huyết lúc đói > 120 mg/dl (0: Không, 1: Có)',
            'restecg': 'Kết quả điện tâm đồ lúc nghỉ (0-2)',
            'thalach': 'Nhịp tim tối đa đạt được (High/Normal/Low)',
            'exang': 'Đau thắt ngực khi tập thể dục (0: Không, 1: Có)',
            'oldpeak': 'ST depression do tập thể dục (High/Normal/Low)',
            'slope': 'Độ dốc của đoạn ST peak (0-2)',
            'ca': 'Số lượng mạch máu chính (0-4)',
            'thal': 'Thalassemia (1-3)'
        }
        
        self.categorical_columns = ['Age', 'trest', 'chol', 'thalach', 'oldpeak']
        self.model = None
        self.label_encoders = {}
        self.feature_names = None
        
    def load_and_prepare_data(self, file_path='Data.csv'):
        """Đọc và chuẩn bị dữ liệu"""
        try:
            data = pd.read_csv(file_path)
            
            # Khởi tạo và lưu trữ các LabelEncoder cho mỗi cột
            for column in self.categorical_columns:
                self.label_encoders[column] = LabelEncoder()
                data[column] = self.label_encoders[column].fit_transform(data[column])
            
            # Tách features và target
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]
            self.feature_names = list(X.columns)
            
            return X, y
        except Exception as e:
            print(f"Lỗi khi đọc dữ liệu: {str(e)}")
            return None, None
    
    def train_model(self, X, y):
        """Huấn luyện mô hình"""
        try:
            # Chia dữ liệu
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Huấn luyện mô hình
            self.model = DecisionTreeClassifier(random_state=42, max_depth=5)
            self.model.fit(X_train, y_train)
            
            # Đánh giá mô hình
            self._evaluate_model(X_train, X_test, y_train, y_test)
            
            # Lưu mô hình
            self._save_model()
            
            return True
        except Exception as e:
            print(f"Lỗi khi huấn luyện mô hình: {str(e)}")
            return False
    
    def _evaluate_model(self, X_train, X_test, y_train, y_test):
        """Đánh giá chi tiết mô hình"""
        # Tính điểm trên tập huấn luyện và kiểm tra
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        
        print("\nĐÁNH GIÁ MÔ HÌNH:")
        print("------------------")
        print(f'Độ chính xác trên tập huấn luyện: {train_score:.2f}')
        print(f'Độ chính xác trên tập kiểm tra: {test_score:.2f}')
        print(f'Độ chính xác cross-validation: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})')
        
        # Tạo confusion matrix
        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        # Vẽ confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Ma trận nhầm lẫn')
        plt.ylabel('Thực tế')
        plt.xlabel('Dự đoán')
        plt.savefig('confusion_matrix.png')
        plt.close()
        
        # In báo cáo phân loại
        print("\nBÁO CÁO PHÂN LOẠI CHI TIẾT:")
        print("----------------------------")
        print(classification_report(y_test, y_pred, target_names=['Không khỏe mạnh', 'Khỏe mạnh']))
        
        # Vẽ và lưu cây quyết định
        self._plot_decision_tree()
        
        # Hiển thị độ quan trọng của đặc trưng
        self._show_feature_importance()
    
    def _plot_decision_tree(self):
        """Vẽ và lưu cây quyết định"""
        plt.figure(figsize=(20,10))
        plot_tree(self.model, feature_names=self.feature_names, 
                 class_names=['Không khỏe mạnh', 'Khỏe mạnh'],
                 filled=True, rounded=True)
        plt.savefig('decision_tree.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _show_feature_importance(self):
        """Hiển thị và vẽ biểu đồ độ quan trọng của đặc trưng"""
        # Tạo DataFrame độ quan trọng
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_,
            'description': [self.feature_descriptions[f] for f in self.feature_names]
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        print("\nĐỘ QUAN TRỌNG CỦA CÁC ĐẶC TRƯNG:")
        print("--------------------------------")
        pd.set_option('display.max_colwidth', None)
        print(feature_importance)
        
        # Vẽ biểu đồ độ quan trọng
        plt.figure(figsize=(12, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title('Độ quan trọng của các đặc trưng')
        plt.savefig('feature_importance.png', bbox_inches='tight')
        plt.close()
    
    def _save_model(self):
        """Lưu mô hình và các encoder"""
        if not os.path.exists('models'):
            os.makedirs('models')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f'models/heart_disease_model_{timestamp}.joblib'
        
        # Lưu mô hình và các encoder
        joblib.dump({
            'model': self.model,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }, model_path)
        
        print(f"\nĐã lưu mô hình tại: {model_path}")
    
    def load_model(self, model_path):
        """Tải mô hình đã lưu"""
        try:
            saved_data = joblib.load(model_path)
            self.model = saved_data['model']
            self.label_encoders = saved_data['label_encoders']
            self.feature_names = saved_data['feature_names']
            return True
        except Exception as e:
            print(f"Lỗi khi tải mô hình: {str(e)}")
            return False
    
    def get_user_input(self, prompt, valid_values=None):
        """Hàm hỗ trợ nhập dữ liệu với kiểm tra giá trị hợp lệ"""
        while True:
            value = input(prompt)
            if valid_values is None:
                return value
            if value in valid_values:
                return value
            print(f"Giá trị không hợp lệ. Vui lòng chọn một trong các giá trị sau: {valid_values}")
    
    def input_patient_data(self):
        """Nhập thông tin bệnh nhân từ người dùng"""
        print("\nNHẬP THÔNG TIN BỆNH NHÂN:")
        print("---------------------------")
        
        inputs = [
            ("Độ tuổi", "Chọn độ tuổi (Adult/MidleAge/Old): ", ['Adult', 'MidleAge', 'Old']),
            ("Giới tính", "Nhập giới tính (0: Nữ, 1: Nam): ", ['0', '1']),
            ("Đau ngực", "Nhập loại đau ngực (0-3): ", ['0', '1', '2', '3']),
            ("Huyết áp", "Nhập huyết áp lúc nghỉ (High/Normal/Low): ", ['High', 'Normal', 'Low']),
            ("Cholesterol", "Nhập mức cholesterol (Normal/High Risk/Extreme): ", ['Normal', 'High Risk', 'Extreme']),
            ("Đường huyết", "Đường huyết lúc đói > 120 mg/dl? (0: Không, 1: Có): ", ['0', '1']),
            ("Điện tâm đồ", "Kết quả điện tâm đồ lúc nghỉ (0-2): ", ['0', '1', '2']),
            ("Nhịp tim", "Nhịp tim tối đa (High/Normal/Low): ", ['High', 'Normal', 'Low']),
            ("Đau thắt ngực khi tập", "Có đau thắt ngực khi tập thể dục? (0: Không, 1: Có): ", ['0', '1']),
            ("ST Depression", "ST depression (High/Normal/Low): ", ['High', 'Normal', 'Low']),
            ("Độ dốc ST", "Độ dốc của đoạn ST peak (0-2): ", ['0', '1', '2']),
            ("Số mạch máu", "Số lượng mạch máu chính (0-4): ", ['0', '1', '2', '3', '4']),
            ("Thalassemia", "Thalassemia (1-3): ", ['1', '2', '3'])
        ]
        
        values = []
        for title, prompt, valid_values in inputs:
            print(f"\n{title}:")
            value = self.get_user_input(prompt, valid_values)
            values.append(int(value) if valid_values[0].isdigit() else value)
        
        return tuple(values)
    
    def predict(self, patient_data):
        """Dự đoán tình trạng sức khỏe tim mạch"""
        try:
            # Tạo DataFrame với tên cột
            input_df = pd.DataFrame([patient_data], columns=self.feature_names)
            
            # Chuyển đổi các giá trị categorical
            for column in self.categorical_columns:
                input_df[column] = self.label_encoders[column].transform(input_df[column])
            
            # Dự đoán
            prediction = self.model.predict(input_df)
            probability = self.model.predict_proba(input_df)
            
            return "Khỏe mạnh" if prediction[0] == 1 else "Không khỏe mạnh", probability[0]
        except Exception as e:
            print(f"Lỗi khi dự đoán: {str(e)}")
            return None, None

def main():
    predictor = HeartDiseasePredictor()
    
    # Kiểm tra xem có mô hình đã lưu không
    if os.path.exists('models'):
        model_files = [f for f in os.listdir('models') if f.endswith('.joblib')]
        if model_files:
            latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join('models', x)))
            print(f"\nĐang tải mô hình đã lưu: {latest_model}")
            predictor.load_model(os.path.join('models', latest_model))
        else:
            print("\nKhông tìm thấy mô hình đã lưu. Đang huấn luyện mô hình mới...")
            X, y = predictor.load_and_prepare_data()
            if X is not None:
                predictor.train_model(X, y)
    else:
        print("\nĐang huấn luyện mô hình mới...")
        X, y = predictor.load_and_prepare_data()
        if X is not None:
            predictor.train_model(X, y)
    
    while True:
        # Nhập dữ liệu từ người dùng
        patient_data = predictor.input_patient_data()
        
        # Dự đoán
        print("\nKẾT QUẢ CHUẨN ĐOÁN:")
        print("--------------------")
        prediction, probabilities = predictor.predict(patient_data)
        
        if prediction is not None:
            print(f"Kết quả: {prediction}")
            print(f"Xác suất: Không khỏe mạnh: {probabilities[0]:.2f}, Khỏe mạnh: {probabilities[1]:.2f}")
        
        # Hỏi người dùng có muốn tiếp tục
        choice = input("\nBạn có muốn tiếp tục chuẩn đoán cho bệnh nhân khác không? (y/n): ")
        if choice.lower() != 'y':
            break

if __name__ == "__main__":
    main() 