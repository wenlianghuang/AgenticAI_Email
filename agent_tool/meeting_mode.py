import subprocess
def meeting_mode():
    try:
        subprocess.run(["C:\\Users\\matt\\Desktop\\Agentic_AI_Tool\\dist\\Set_Brightness.exe"],check=True)

        #subprocess.run(["C:\\Users\\matt\\Desktop\\Agentic_AI_Tool\\dist\\Check_Internet.exe"],check=True)
        subprocess.run(["C:\\Users\\matt\\Desktop\\Agentic_AI_Tool\\dist\\Vedio_Recording.exe"],check=True)
        return "✅ Start Meeting Model successfully."
    except subprocess.CalledProcessError as e:
        return f"❌ Failed to start Meeting Model: {str(e)}"
    except Exception as e:
        return f"❌ An unexpected error occurred: {str(e)}"