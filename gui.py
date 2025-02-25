import sys
from PyQt6.QtWidgets import ( QMainWindow, QWidget, QLabel,
                             QComboBox, QVBoxLayout,  QPushButton,
                             QGridLayout, )
from PyQt6.QtCore import Qt
from predictor import HealthPredictor


class HealthPredictorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.predictor = HealthPredictor()
        self.rules_df = self.predictor.load_rules_from_csv()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Heart Disease Predictor')
        self.setGeometry(100, 100, 800, 600)

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Create grid layout for input fields
        grid_layout = QGridLayout()

        # Define input fields
        self.inputs = {}

        # Age options
        age_options = ['Adult', 'MidleAge', 'Old']
        self.create_input_field('Age', age_options, grid_layout, 0)

        # Sex options
        sex_options = [0, 1]
        self.create_input_field('sex', sex_options, grid_layout, 1)

        # Chest Pain Type options
        cp_options = [0, 1, 2, 3]
        self.create_input_field('cp', cp_options, grid_layout, 2)

        # Resting Blood Pressure options
        trest_options = ['High','Low','Normal']
        self.create_input_field('trest', trest_options, grid_layout, 3)

        # Cholesterol options
        chol_options = ['Extreme', 'High Risk', 'Normal']
        self.create_input_field('chol', chol_options, grid_layout, 4)

        # Fasting Blood Sugar options
        fbs_options = [0, 1]
        self.create_input_field('fbs', fbs_options, grid_layout, 5)

        # Resting ECG options
        restecg_options = [0, 1, 2]
        self.create_input_field('restecg', restecg_options, grid_layout, 6)

        # Maximum Heart Rate options
        thalach_options = ['High','Low','Normal']
        self.create_input_field('thalach', thalach_options, grid_layout, 7)

        # Exercise Induced Angina options
        exang_options = [0, 1]
        self.create_input_field('exang', exang_options, grid_layout, 8)

        # ST Depression options
        oldpeak_options = ['High','Low','Normal']
        self.create_input_field('oldpeak', oldpeak_options, grid_layout, 9)

        # Slope options
        slope_options = [0, 1, 2]
        self.create_input_field('slope', slope_options, grid_layout, 10)

        # Number of Major Vessels options
        ca_options = [0, 1, 2, 3, 4]
        self.create_input_field('ca', ca_options, grid_layout, 11)

        # Thalassemia options
        thal_options = [0, 1, 2, 3]
        self.create_input_field('thal', thal_options, grid_layout, 12)

        main_layout.addLayout(grid_layout)

        # Create predict button
        predict_button = QPushButton('Predict', self)
        predict_button.clicked.connect(self.predict)
        predict_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px;
                font-size: 16px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        main_layout.addWidget(predict_button)

        # Create result label
        self.result_label = QLabel('')
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                padding: 10px;
                margin-top: 10px;
                width: 300px;
            }
        """)
        main_layout.addWidget(self.result_label)

    def create_input_field(self, name, options, layout, row):
        label = QLabel(f'{name}:')
        combo = QComboBox()
        combo.addItems([str(opt) for opt in options])
        layout.addWidget(label, row, 0)
        layout.addWidget(combo, row, 1)
        self.inputs[name] = combo

    def predict(self):
        # Get values from input fields and convert to appropriate format
        patient_data = {}
        for attr, combo in self.inputs.items():
            value = combo.currentText()
            try:
                value = int(value)
            except ValueError:
                pass
            patient_data[attr] = value
        print(patient_data)
        # Get prediction
        result = self.predictor.diagnose(patient_data, self.rules_df)

        # Display result
        result_text = f"""
        Diagnosis: {result['diagnosis']}
        Rule ID: {result['rule_id']}
        Matching Conditions: {result['matching_conditions']}
        """


        # Update result label
        self.result_label.setText(result_text)
        if result['diagnosis'] == 'Bệnh tim':
            self.result_label.setStyleSheet("QLabel { color: red; }")
        else:
            self.result_label.setStyleSheet("QLabel { color: green; }")