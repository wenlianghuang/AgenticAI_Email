import subprocess
def recording_vedio():
    try:
        subprocess.run(["C:\\Users\\matt\\Desktop\\Agentic_AI_Tool\\dist\\Open_Camera_Microphone.exe"],check=True)
        return "✅ Start Vedio Recording successfully."
    except subprocess.CalledProcessError as e:
        return f"❌ Failed to start Meeting Model: {str(e)}"
    except Exception as e:
        return f"❌ An unexpected error occurred: {str(e)}"