import whisper
from transformers import pipeline
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Function to transcribe audio using Whisper
def transcribe_audio(audio_file):
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    return result["text"]

# Function to extract medical terms using Medical BERT
def extract_medical_terms(text):
    nlp = pipeline("ner", model="cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    entities = nlp(text)
    return list(set(entity["word"] for entity in entities))  # Remove duplicates

# Function to summarize the medical report using Gemini API
def summarize_report(report_text):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(f"Summarize the following medical report:\n\n{report_text}")
    return response.text.strip()

# Function to save the summary in a .txt file
def save_summary(patient_name, transcription, medical_terms, summary):
    txt_filename = f"{patient_name}_summary.txt"
    with open(txt_filename, "w") as txt_file:
        txt_file.write(f"Patient Name: {patient_name}\n\n")
        txt_file.write("Transcription:\n")
        txt_file.write(transcription + "\n\n")
        txt_file.write("Extracted Medical Terms:\n")
        txt_file.write(", ".join(medical_terms) + "\n\n")
        txt_file.write("Summary:\n")
        txt_file.write(summary + "\n")
    
    print(f"\n✅ Summary saved as: {txt_filename}")

# Main function to execute the CLI workflow
def main():
    audio_path = input("Enter the audio file path: ")
    patient_name = input("Enter the patient's name: ")

    print("\n🔄 Transcribing...")
    transcription = transcribe_audio(audio_path)
    print("✅ Transcription Completed.")

    print("\n🔄 Extracting Medical Terms...")
    medical_terms = extract_medical_terms(transcription)
    print(f"✅ Medical Terms Found: {medical_terms}")

    print("\n🔄 Summarizing the Report...")
    report_text = f"Patient: {patient_name}\nTranscription: {transcription}\nExtracted Medical Terms: {', '.join(medical_terms)}"
    summary = summarize_report(report_text)
    print(f"✅ Summary Generated:\n{summary}")

    print("\n💾 Saving the summary...")
    save_summary(patient_name, transcription, medical_terms, summary)
    print("✅ Summary saved successfully.")

if __name__ == "__main__":
    main()
