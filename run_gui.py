# FX-SLV Dashboard Main Application
# Run this file to launch the FX-SLV Dashboard GUI with Streamlit

import sys
from streamlit.web import cli as stcli

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", "GUI/main_tab.py"]
    sys.exit(stcli.main())
