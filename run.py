import os, subprocess  
os.chdir(r"C:\Users\M RAMA MANI\Desktop\border_ai_sih")  
subprocess.run(["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]) 
