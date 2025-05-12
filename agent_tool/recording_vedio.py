import subprocess
import os
def recording_video(filename: str) -> str:
    """
    Start video recording using the specified filename.
    """
    exe_path = "C:\\Users\\matt\\Desktop\\Agentic_AI_Tool\\dist\\Open_Camera_Microphone.exe"
    exe_dir = os.path.dirname(exe_path)
    print(f"DEBUG: filename = {filename}")  # Debug message
    try:
        #subprocess.run(["C:\\Users\\matt\\Desktop\\Agentic_AI_Tool\\dist\\Open_Camera_Microphone.exe", "--output", filename], check=True)
        subprocess.run([exe_path, "--output", filename], check=True, cwd=exe_dir)
        return "✅ Start Vedio Recording successfully."
    except subprocess.CalledProcessError as e:
        return f"❌ Failed to start Meeting Model: {str(e)}"
    except Exception as e:
        return f"❌ An unexpected error occurred: {str(e)}"