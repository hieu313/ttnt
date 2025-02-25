import sys
from PyQt6.QtWidgets import QApplication
from gui import HealthPredictorGUI

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Use Fusion style for a modern look
    window = HealthPredictorGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()