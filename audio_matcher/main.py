# main.py
import sys
from PyQt6.QtWidgets import QApplication, QMessageBox

# Pokusíme se importovat hlavní aplikaci. 
# Pokud to Defender zařízne (typicky na matplotlib/Pillow), chytíme to.
try:
    from gui.app import AudioMatcherApp
except ImportError as e:
    # Vytvoříme provizorní Qt Aplikaci jen pro zobrazení chybové hlášky
    app = QApplication(sys.argv)
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Icon.Critical)
    msg.setWindowTitle("Chyba zabezpečení Windows")
    msg.setText("Systém Windows zablokoval načtení grafické knihovny.")
    msg.setInformativeText(
        f"Detail chyby:\n{e}\n\n"
    )
    msg.exec()
    sys.exit(1)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AudioMatcherApp()
    window.show()
    sys.exit(app.exec())