import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pydantic import BaseModel

# Define the input schema for the send_email_tool
# This ensures that the tool receives the correct input format
class SendEmailInput(BaseModel):
    recipient: str
    subject: str
    body: str

# Function to send an email
# Parameters:
# - recipient: The email address of the recipient
# - subject: The subject of the email
# - body: The body content of the email
def send_email(recipient: str, subject: str, body: str):
    # Sender email and password (retrieved from environment variables)
    sender_email = "wenliangmatt@gmail.com"
    sender_password = os.getenv('wenliangmattapppwd')  # App-specific password for Gmail
    #print(f"DEBUG: sender_password = {sender_password}")  # Debug message
    smtp_server = "smtp.gmail.com"
    smtp_port = 587

    try:
        # Create the email message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        # Connect to the SMTP server and send the email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()  # Start TLS encryption
            server.login(sender_email, sender_password)  # Login to the SMTP server
            server.send_message(msg)  # Send the email

        #print("Email sent successfully!")
        return "Email sent successfully! STOP"
    except Exception as e:
        #print(f"Failed to send email: {e}")
        return f"Failed to send email: {e}. STOP!"