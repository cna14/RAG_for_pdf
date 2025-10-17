import google.generativeai as genai
import os

# Thay thế bằng API key của bạn, hoặc đảm bảo file .env đúng
# from dotenv import load_dotenv
# load_dotenv()
# api_key = os.getenv("GOOGLE_API_KEY")

api_key = "AIzaSyCEXYVjb8ZQtmr47EfdzuBCcm3flhSnx_c" # Dán key của bạn vào đây để kiểm tra nhanh

genai.configure(api_key=api_key)

print("Các model hỗ trợ phương thức 'generateContent':")
for m in genai.list_models():
  if 'generateContent' in m.supported_generation_methods:
    print(f"- {m.name}")