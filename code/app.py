import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from ui.leap_labs_adversarial_attack import Ui_MainWindow  # Import the generated UI class
from utils import  classify_mnist

class MainApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.initialize_ui()

    def initialize_ui(self):
        """Set up the UI and initialize variables."""
        self.ui = Ui_MainWindow()  # Create an instance of the UI class
        self.ui.setupUi(self)  # Set up the UI on this QMainWindow instance

        # Attributes to store variables
        self.selected_file = None  

    #     # Connect buttons to functions
        self.ui.upload_image.clicked.connect(self.upload_image_action)
        self.ui.run_MNIST_classification.clicked.connect(self.run_MNIST_classification_action)
        self.ui.run_adversarial_attack_and_rerun_MNIST_classification.clicked.connect(self.run_adversarial_attack_and_rerun_MNIST_classification_action)
        self.ui.reset_button.clicked.connect(self.reset_action)  

    def upload_image_action(self):
        # Open a file dialog to select a file
        file_name, _ = QFileDialog.getOpenFileName(None, "Select File", "", "All Files (*.*);;Text Files (*.txt);;FASTA Files (*.fasta)")

        if file_name:  # If a file was selected
            QMessageBox.information(None, "File Selected", f"You selected: {file_name}")
            # You can use the selected file path (file_name) for further processing
            self.selected_file = file_name  # Store the file path
            print(f"Selected file: {file_name}")
        else:
            QMessageBox.warning(None, "No File", "No file was selected.")
    
    
    def run_MNIST_classification_action(self):
        '''function for downstream workflow when run MNIST classification button is pressed'''
        #message to start function
        QMessageBox.information(None, "Running MNIST classification", "Click OK to begin classification.")
        
        #Load up variables from the class space
        image_path = self.selected_file

        #Run MNIST classification function
        classification = classify_mnist(image_path, model_path='models/lenet_mnist_model.pth')
        message = f"Class identified: {classification}. Click OK to continue."
        QMessageBox.information(None, "Classification assigned", message)

    def run_adversarial_attack_and_rerun_MNIST_classification_action(self):
        '''function for downstream workflow when run adversarial attack and rerun MNIST classification button is pressed'''
        #message to start function
        QMessageBox.information(None, "Running adversarial attack and rerunning MNIST classification", "Sorry not implemented yet!")

    def reset_action(self):
        """Reset the entire application to its initial state."""
        # Remove the current UI
        self.centralWidget().deleteLater()
        # Reinitialize everything
        self.initialize_ui()

        

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainApp()
    main_window.show()
    sys.exit(app.exec_())