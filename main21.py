# Import necessary libraries
import os
import json
import datetime
import pandas as pd
import torch
from transformers import pipeline
from langdetect import detect
from deep_translator import GoogleTranslator
import google.generativeai as genai
import speech_recognition as sr
import pyaudio
import wave
import numpy as np
from time import sleep
import threading
import queue
import io
import tempfile

# Available languages with their codes
LANGUAGES = {
    "1": {"name": "English", "code": "en", "speech_code": "en-US"},
    "2": {"name": "Hindi", "code": "hi", "speech_code": "hi-IN"},
    "3": {"name": "Spanish", "code": "es", "speech_code": "es-ES"},
    "4": {"name": "French", "code": "fr", "speech_code": "fr-FR"},
    "5": {"name": "German", "code": "de", "speech_code": "de-DE"},
    "6": {"name": "Chinese", "code": "zh-CN", "speech_code": "zh-CN"},
    "7": {"name": "Arabic", "code": "ar", "speech_code": "ar-AE"},
    "8": {"name": "Russian", "code": "ru", "speech_code": "ru-RU"},
    "9": {"name": "Portuguese", "code": "pt", "speech_code": "pt-BR"},
    "10": {"name": "Bengali", "code": "bn", "speech_code": "bn-IN"},
    "11": {"name": "Japanese", "code": "ja", "speech_code": "ja-JP"},
    "12": {"name": "Korean", "code": "ko", "speech_code": "ko-KR"},
    "13": {"name": "Tamil", "code": "ta", "speech_code": "ta-IN"},
    "14": {"name": "Telugu", "code": "te", "speech_code": "te-IN"},
    "15": {"name": "Marathi", "code": "mr", "speech_code": "mr-IN"}
}

# Additional language codes for reference
ADDITIONAL_LANGS = {
    "Urdu": {"code": "ur", "speech_code": "ur-PK"}, 
    "Punjabi": {"code": "pa", "speech_code": "pa-IN"}, 
    "Gujarati": {"code": "gu", "speech_code": "gu-IN"}, 
    "Malayalam": {"code": "ml", "speech_code": "ml-IN"}, 
    "Kannada": {"code": "kn", "speech_code": "kn-IN"}, 
    "Odia": {"code": "or", "speech_code": "or-IN"}, 
    "Assamese": {"code": "as", "speech_code": "as-IN"}, 
    "Thai": {"code": "th", "speech_code": "th-TH"}, 
    "Vietnamese": {"code": "vi", "speech_code": "vi-VN"}, 
    "Indonesian": {"code": "id", "speech_code": "id-ID"}, 
    "Malay": {"code": "ms", "speech_code": "ms-MY"}, 
    "Turkish": {"code": "tr", "speech_code": "tr-TR"},
    "Italian": {"code": "it", "speech_code": "it-IT"}, 
    "Dutch": {"code": "nl", "speech_code": "nl-NL"}, 
    "Swedish": {"code": "sv", "speech_code": "sv-SE"}, 
    "Polish": {"code": "pl", "speech_code": "pl-PL"},
    "Ukrainian": {"code": "uk", "speech_code": "uk-UA"}, 
    "Greek": {"code": "el", "speech_code": "el-GR"}, 
    "Hebrew": {"code": "he", "speech_code": "he-IL"}, 
    "Persian": {"code": "fa", "speech_code": "fa-IR"}
}

#cell4
# Translation functions with better error handling
def safe_translate(text, source_lang, target_lang):
    """Safely translate text between languages with fallbacks"""
    if not text or source_lang == target_lang:
        return text
        
    try:
        # Try with specified source language
        translator = GoogleTranslator(source=source_lang, target=target_lang)
        translated = translator.translate(text)
        
        # If translation failed or returned None/empty
        if not translated:
            raise Exception("Empty translation result")
            
        return translated
    except Exception as e:
        print(f"First translation attempt failed: {e}")
        try:
            # Try with auto-detection as fallback
            translator = GoogleTranslator(source='auto', target=target_lang)
            translated = translator.translate(text)
            if translated:
                return translated
            else:
                return text
        except Exception as e2:
            print(f"Fallback translation failed: {e2}")
            return text

# Improved language detection function that avoids detecting on proper names
def detect_language_safely(text):
    """Detect language with fallback to English and proper name handling"""
    if not text or len(text.strip()) < 5:  # Too short for reliable detection
        return "en"
    
    # Skip detection for likely proper names (single words with capitalization)
    words = text.strip().split()
    if len(words) == 1 and words[0][0].isupper():
        return "en"  # Assume proper names are in the user's selected language
        
    try:
        detected = detect(text)
        return detected
    except:
        return "en"
#cell5
# Patient data management
class PatientDatabase:
    def __init__(self, file_path="patient_records.json"):
        self.file_path = file_path
        self.records = self.load_records()
        
    def load_records(self):
        try:
            if os.path.exists(self.file_path):
                with open(self.file_path, 'r', encoding='utf-8') as file:
                    return json.load(file)
            return {}
        except Exception as e:
            print(f"Error loading records: {e}")
            return {}
    
    def save_records(self):
        try:
            with open(self.file_path, 'w', encoding='utf-8') as file:
                json.dump(self.records, file, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving records: {e}")
    
    def add_patient(self, patient_id, data):
        if patient_id not in self.records:
            self.records[patient_id] = []
        
        # Add timestamp to the consultation
        data["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.records[patient_id].append(data)
        self.save_records()
    
    def get_patient_history(self, patient_id):
        return self.records.get(patient_id, [])
    
#cell6
# Voice Assistant class with multilingual support
class VoiceAssistant:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.audio_queue = queue.Queue()
        self.is_recording = False
        
    def record_audio(self):
        # Audio recording parameters
        CHUNK = 1024
        FORMAT = pyaudio.paFloat32
        CHANNELS = 1
        RATE = 44100
        
        p = pyaudio.PyAudio()
        
        stream = p.open(format=FORMAT,
                       channels=CHANNELS,
                       rate=RATE,
                       input=True,
                       frames_per_buffer=CHUNK)
        
        print("\nRecording... Press Enter to stop")
        self.is_recording = True
        frames = []
        
        # Start recording thread
        def record():
            while self.is_recording:
                data = stream.read(CHUNK)
                frames.append(data)
                
        # Start input thread
        def wait_for_input():
            input()  # Wait for Enter key
            self.is_recording = False
            
        record_thread = threading.Thread(target=record)
        input_thread = threading.Thread(target=wait_for_input)
        
        record_thread.start()
        input_thread.start()
        
        # Wait for recording to finish
        record_thread.join()
        input_thread.join()
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        return frames, RATE
        
    def transcribe_audio(self, frames, rate, language_code="en-US"):
        """Transcribe audio with support for multiple languages"""
        # Convert frames to audio data
        audio_data = b''.join(frames)
        audio_array = np.frombuffer(audio_data, dtype=np.float32)
        
        # Convert float32 audio to int16 (required for WAV)
        audio_array_int = (audio_array * 32767).astype(np.int16)
        
        # Create a temporary WAV file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_wav:
            with wave.open(temp_wav.name, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 2 bytes for int16
                wf.setframerate(rate)
                wf.writeframes(audio_array_int.tobytes())
        
        # Use speech recognition with the temporary file
        try:
            with sr.AudioFile(temp_wav.name) as source:
                audio = self.recognizer.record(source)
                # Use the specified language for recognition
                text = self.recognizer.recognize_google(audio, language=language_code)
                
            # Clean up the temporary file
            os.unlink(temp_wav.name)
            return text
            
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError:
            return "Could not request results"
        finally:
            # Ensure temp file is deleted even if an error occurs
            if os.path.exists(temp_wav.name):
                os.unlink(temp_wav.name)
                
    # Map language codes to Google Speech Recognition language codes
    def get_speech_recognition_code(self, lang_code):
        """Convert ISO language code to Google Speech Recognition language code"""
        # Mapping of common language codes
        speech_codes = {
            "en": "en-US",
            "hi": "hi-IN",
            "es": "es-ES",
            "fr": "fr-FR",
            "de": "de-DE",
            "zh-CN": "zh-CN",
            "ar": "ar-AE",
            "ru": "ru-RU",
            "pt": "pt-BR",
            "bn": "bn-IN",
            "ja": "ja-JP",
            "ko": "ko-KR",
            "ta": "ta-IN",
            "te": "te-IN",
            "mr": "mr-IN"
        }
        
        return speech_codes.get(lang_code, "en-US")  # Default to en-US if not found
#cell7
# Virtual Doctor class with updated error handling for API
class VirtualDoctor:
    def __init__(self, gemini_model):
        self.model = gemini_model
        self.db = PatientDatabase()
        try:
            self.sentiment_analyzer = pipeline("sentiment-analysis")
        except Exception as e:
            print(f"Sentiment analyzer initialization failed: {e}")
            self.sentiment_analyzer = None
        
    def get_medical_response(self, symptoms, patient_history=None):
        """Generate medical response based on symptoms with improved error handling"""
        history_text = "No previous records"
        if patient_history:
            history_text = "\n".join([
                f"Date: {record.get('timestamp', 'Unknown')}, " +
                f"Symptoms: {record.get('symptoms', 'None')}, " +
                f"Diagnosis: {record.get('response', 'None')[:100]}..."
                for record in patient_history[:3]  # Last 3 consultations
            ])
        
        prompt = f"""
        Act as a medical assistant providing preliminary advice. 
        The patient has reported the following symptoms: {symptoms}
        
        Patient history: {history_text}
        
        Provide:
        1. Possible conditions (with clear disclaimer that this is not a diagnosis)
        2. Recommendations for home care if appropriate
        3. Clear advice on when to seek professional medical help
        4. Any follow-up questions that would help clarify the condition
        
        Keep responses informative but cautious, and always prioritize patient safety.
        """
        
        try:
            # Updated to handle potential API changes
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg or "not found" in error_msg:
                return "I'm experiencing technical difficulties with the AI service. This may be due to an outdated model name or API version. Please contact the developer to update the application with the latest Gemini API specifications."
            return f"I'm having trouble generating a response. Please try again later. Error: {str(e)}"

#cell8
# User interface functions
def select_language():
    """Let user select their preferred language"""
    print("Welcome to Virtual Doctor Assistant")
    print("Please select your language:")
    
    for key, lang in LANGUAGES.items():
        print(f"{key}. {lang['name']}")
    
    print("Enter the number of your language choice (or type 'other'): ", end="")
    choice = input().strip()
    
    if choice in LANGUAGES:
        return LANGUAGES[choice]["code"], LANGUAGES[choice]["name"]
    elif choice.lower() == "other":
        print("Please enter your language code (e.g., 'ja' for Japanese): ", end="")
        custom_code = input().strip().lower()
        print("Please enter your language name: ", end="")
        custom_name = input().strip()
        return custom_code, custom_name
    else:
        # Try to interpret the choice as a language code
        for lang_name, lang_info in ADDITIONAL_LANGS.items():
            if choice.lower() == lang_info["code"]:
                return lang_info["code"], lang_name
        
        # If all else fails, try to detect language from further input
        print("I couldn't recognize that choice. Please type a sentence in your preferred language:")
        sample_text = input().strip()
        detected_code = detect_language_safely(sample_text)
        return detected_code, f"Detected language ({detected_code})"

def collect_patient_info(lang_code):
    """Collect basic patient information - FIXED to avoid language detection on names"""
    info = {}
    
    questions = {
        "name": "Please enter your name: ",
        "age": "Please enter your age: ",
        "gender": "Please enter your gender (Male/Female/Other): ",
        "phone": "Please enter your phone number (optional): "
    }
    
    for key, question in questions.items():
        translated_question = safe_translate(question, "en", lang_code)
        print(translated_question, end="")
        response = input().strip()
        
        # Do NOT perform language detection on personal information
        info[key] = response
    
    # Return the original language code without changing it
    return info, lang_code

#cell9
# Initialize Gemini API with latest model names
def initialize_gemini(api_key):
    genai.configure(api_key=api_key)
    # Updated to use gemini-2.0-flash model which is available as of March 2025
    return genai.GenerativeModel('gemini-2.0-flash')

#cell10
# Combined main application with text and voice options and improved API key handling
def run_virtual_doctor():
    # Use a pre-defined API key instead of asking from the user
    api_key = r"API key"  # Developer should replace this with their actual API key
    
    try:
        # Initialize Gemini
        model = initialize_gemini(api_key)
        
        # Ask user for input method preference
        print("\nHow would you like to interact with the Virtual Doctor?")
        print("1. Text input (supports multiple languages)")
        print("2. Voice input (supports multiple languages)")
        print("Enter your choice (1 or 2): ", end="")
        
        input_choice = input().strip()
        
        if input_choice == "2":
            # Voice input mode (now with multilingual support)
            run_voice_doctor(model)
        else:
            # Default to text input mode
            run_text_doctor(model)
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please try again later.")


#cell11
# Text-based doctor function
def run_text_doctor(model):
    try:
        # Select language
        lang_code, lang_name = select_language()
        print(f"\nSelected language: {lang_name} ({lang_code})")
        
        # Initialize virtual doctor
        doctor = VirtualDoctor(model)
        
        # Translate welcome message
        welcome_msg = "Welcome to the Virtual Doctor Assistant. I'm here to help with your health concerns."
        translated_welcome = safe_translate(welcome_msg, "en", lang_code)
        print(f"\n{translated_welcome}")
        
        # Collect patient information
        patient_info_msg = "First, I need to collect some basic information."
        translated_info_msg = safe_translate(patient_info_msg, "en", lang_code)
        print(f"\n{translated_info_msg}")
        
        # Get patient info without language detection on names
        patient_info, _ = collect_patient_info(lang_code)
        
        # Use original language code, don't update based on name input
        patient_id = f"{patient_info['name']}_{patient_info.get('phone', 'unknown')}"
        
        # Get patient history
        patient_history = doctor.db.get_patient_history(patient_id)
        
        # Collect symptoms
        symptoms_prompt = "Please describe your symptoms or health concerns in detail:"
        translated_symptoms_prompt = safe_translate(symptoms_prompt, "en", lang_code)
        print(f"\n{translated_symptoms_prompt}")
        
        symptoms_input = input().strip()
        
        # Only detect language on longer text like symptoms
        if len(symptoms_input.split()) > 3:  # Only detect if more than 3 words
            detected_lang = detect_language_safely(symptoms_input)
            if detected_lang != lang_code and detected_lang != "en":
                lang_detect_msg = f"I detected that you're writing in a different language. I'll continue in this language."
                translated_detect_msg = safe_translate(lang_detect_msg, "en", detected_lang)
                print(f"\n{translated_detect_msg}")
                lang_code = detected_lang
        
        # Translate symptoms to English for processing
        english_symptoms = safe_translate(symptoms_input, lang_code, "en")
        
        # Generate medical response
        medical_response = doctor.get_medical_response(english_symptoms, patient_history)
        
        # Translate response back to user's language
        translated_response = safe_translate(medical_response, "en", lang_code)
        print(f"\n{translated_response}")
        
        # Save consultation data
        consultation_data = {
            "symptoms": english_symptoms,
            "original_symptoms": symptoms_input,
            "response": medical_response,
            "translated_response": translated_response,
            "language": lang_code,
            "patient_info": patient_info,
            "input_method": "text"
        }
        
        doctor.db.add_patient(patient_id, consultation_data)
        
        # Ask if user wants follow-up questions
        followup_msg = "\nDo you have any follow-up questions? (yes/no)"
        translated_followup = safe_translate(followup_msg, "en", lang_code)
        print(translated_followup)
        
        followup_response = input().strip().lower()
        followup_response_en = safe_translate(followup_response, lang_code, "en").lower()
        
        if "yes" in followup_response_en or "y" == followup_response_en:
            followup_prompt = "What else would you like to know?"
            translated_followup_prompt = safe_translate(followup_prompt, "en", lang_code)
            print(f"\n{translated_followup_prompt}")
            
            followup_input = input().strip()
            english_followup = safe_translate(followup_input, lang_code, "en")
            
            # Generate follow-up response
            followup_context = f"Previous symptoms: {english_symptoms}\nFollow-up question: {english_followup}"
            followup_response = doctor.get_medical_response(followup_context, patient_history)
            
            # Translate follow-up response
            translated_followup_response = safe_translate(followup_response, "en", lang_code)
            print(f"\n{translated_followup_response}")
            
            # Save follow-up data
            doctor.db.add_patient(patient_id, {
                "followup_question": english_followup,
                "followup_response": followup_response,
                "language": lang_code,
                "input_method": "text"
            })
        
        # Closing message
        closing_msg = "\nThank you for using Virtual Doctor Assistant. Remember, this is not a replacement for professional medical advice. Please consult a healthcare provider for proper diagnosis and treatment."
        translated_closing = safe_translate(closing_msg, "en", lang_code)
        print(translated_closing)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please try again later.")
#cell12
# Voice-based doctor function with multilingual support
def run_voice_doctor(model):
    try:
        # Initialize voice assistant
        voice = VoiceAssistant()
        
        # Select language
        lang_code, lang_name = select_language()
        print(f"\nSelected language: {lang_name} ({lang_code})")
        
        # Get the appropriate speech recognition language code
        speech_lang_code = voice.get_speech_recognition_code(lang_code)
        
        # Initialize virtual doctor
        doctor = VirtualDoctor(model)
        
        # Translate welcome message
        welcome_msg = "Welcome to the Voice-based Virtual Doctor Assistant. I'll help assess your health concerns through voice interaction."
        translated_welcome = safe_translate(welcome_msg, "en", lang_code)
        print(f"\n{translated_welcome}")
        
        # Collect patient information
        patient_info_msg = "First, I need to collect some basic information."
        translated_info_msg = safe_translate(patient_info_msg, "en", lang_code)
        print(f"\n{translated_info_msg}")
        
        # Get patient info without language detection on names
        patient_info, _ = collect_patient_info(lang_code)
        
        patient_id = f"{patient_info['name']}_{patient_info.get('phone', 'unknown')}"
        
        # Get patient history
        patient_history = doctor.db.get_patient_history(patient_id)
        
        # Collect symptoms through voice
        symptoms_prompt = "Please describe your symptoms or health concerns when recording starts."
        translated_symptoms_prompt = safe_translate(symptoms_prompt, "en", lang_code)
        print(f"\n{translated_symptoms_prompt}")
        
        # Record and transcribe audio in the selected language
        frames, rate = voice.record_audio()
        symptoms_input = voice.transcribe_audio(frames, rate, speech_lang_code)
        
        # Show transcribed text to user
        transcribed_msg = f"Transcribed symptoms: {symptoms_input}"
        translated_transcribed = safe_translate(transcribed_msg, "en", lang_code)
        print(f"\n{translated_transcribed}")
        
        # Translate symptoms to English for processing if not already in English
        english_symptoms = safe_translate(symptoms_input, lang_code, "en")
        
        # Generate medical response
        medical_response = doctor.get_medical_response(english_symptoms, patient_history)
        
        # Translate response back to user's language
        translated_response = safe_translate(medical_response, "en", lang_code)
        print(f"\n{translated_response}")
        
        # Save consultation data
        consultation_data = {
            "symptoms": english_symptoms,
            "original_symptoms": symptoms_input,
            "response": medical_response,
            "translated_response": translated_response,
            "language": lang_code,
            "patient_info": patient_info,
            "input_method": "voice"
        }
        
        doctor.db.add_patient(patient_id, consultation_data)
        
        # Ask if user wants follow-up questions
        followup_msg = "Do you have any follow-up questions? (yes/no)"
        translated_followup = safe_translate(followup_msg, "en", lang_code)
        print(f"\n{translated_followup}")
        
        followup_response = input().strip().lower()
        followup_response_en = safe_translate(followup_response, lang_code, "en").lower()
        
        if "yes" in followup_response_en or "y" == followup_response_en:
            followup_prompt = "Please ask your follow-up question when recording starts."
            translated_followup_prompt = safe_translate(followup_prompt, "en", lang_code)
            print(f"\n{translated_followup_prompt}")
            
            # Record and transcribe follow-up in the selected language
            frames, rate = voice.record_audio()
            followup_input = voice.transcribe_audio(frames, rate, speech_lang_code)
            
            # Show transcribed follow-up to user
            transcribed_followup = f"Transcribed follow-up: {followup_input}"
            translated_transcribed_followup = safe_translate(transcribed_followup, "en", lang_code)
            print(f"\n{translated_transcribed_followup}")
            
            # Translate follow-up to English for processing
            english_followup = safe_translate(followup_input, lang_code, "en")
            
            # Generate follow-up response
            followup_context = f"Previous symptoms: {english_symptoms}\nFollow-up question: {english_followup}"
            followup_response = doctor.get_medical_response(followup_context, patient_history)
            
            # Translate follow-up response back to user's language
            translated_followup_response = safe_translate(followup_response, "en", lang_code)
            print(f"\n{translated_followup_response}")
            
            # Save follow-up data
            doctor.db.add_patient(patient_id, {
                "followup_question": english_followup,
                "original_followup": followup_input,
                "followup_response": followup_response,
                "translated_followup_response": translated_followup_response,
                "language": lang_code,
                "input_method": "voice"
            })
        
        # Closing message
        closing_msg = "Thank you for using Virtual Doctor Assistant. Remember, this is not a replacement for professional medical advice. Please consult a healthcare provider for proper diagnosis and treatment."
        translated_closing = safe_translate(closing_msg, "en", lang_code)
        print(f"\n{translated_closing}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please try again later.")
#cell13# Run the application
run_virtual_doctor()
