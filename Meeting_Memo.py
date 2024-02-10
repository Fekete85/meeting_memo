import torch
import pandas as pd
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
from pyannote.audio import Pipeline
import tkinter as tk
from tkinter import filedialog
import os
import srt
import sys

### File select ###

def file_select() -> str:
    """
    Opens a file dialog window, allows the user to select a file, and returns the selected file name.

    Returns:
        str: The selected file name.
    """
    root = tk.Tk()
    root.withdraw()
    file_name = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    root.destroy()

    if file_name == "":
        print("No file selected. Exiting program.")
        sys.exit()

    return file_name

# Storeing the file location
file_loc = file_select()
folder_loc, file_name = os.path.split(file_loc)
file_name, file_ext = os.path.splitext(file_loc)

### File select ###
### Loading openai Whisper model ###

device = "cuda:0"  if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]

### Loading openai Whisper model ###
### Generating transcript ###

result = pipe(file_loc, generate_kwargs={"language": "hungarian"}, return_timestamps=True) # Hungarian language model selected

### Generating transcript ###
### Speaker diarization ###

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="HF_Token") # Use your Hugging Face token here from https://hf.co/settings/tokens
pipeline.to(torch.device("cuda"))

diarization = pipeline(file_loc)

### Speaker diarization ###

def get_transcription(result):
    """
    Returns a concatenated transcription of the text chunks in the result.

    Args:
        result (dict): A dictionary containing a list of text chunks.

    Returns:
        str: A concatenated transcription of the text chunks in the result.
    
    Example:
        result = {
            "chunks": [
                {"text": "Hello"},
                {"text": "world"}
            ]
        }
        transcription = get_transcription(result)
        print(transcription)
        
        Output:
        Hello
        world
    """
    transcription = ""
    for chunk in result["chunks"]:
        transcription += chunk["text"] + "\n"
    return transcription

def preprocess_time(time_str):
    """
    Extracts the start and end times from a timestamp string.

    Args:
        time_str (str): A string representing the timestamp in the format "start_time, end_time".

    Returns:
        tuple: A tuple containing the start time (float) and end time (float) extracted from the timestamp string.
    
    Example:
        time_str = "1.5, 2.5"
        start_time, end_time = preprocess_time(time_str)
        print(start_time)  # Output: 1.5
        print(end_time)  # Output: 2.5
    """
    start_time = time_str.split(",")[0]
    end_time = time_str.split(",")[1]
    start_time = float(start_time[1:])
    end_time = float(end_time[:-1])
    return start_time, end_time

def result_to_dataframe(result):
    """
    Convert a dictionary into a pandas DataFrame.

    Args:
        result (dict): A dictionary containing a list of chunks, where each chunk has a timestamp and text.

    Returns:
        df (DataFrame): A pandas DataFrame containing the start time, end time, and text of each chunk in the result dictionary.
    """
    df = pd.DataFrame(columns=["start", "end", "text"])
    for chunk in result["chunks"]:
        timestampstr = str(chunk["timestamp"])
        start_time, end_time = preprocess_time(timestampstr)
        df = df._append(
            {
                "start": start_time,
                "end": end_time,
                "text": chunk["text"],
            },
            ignore_index=True,
        )
    return df

transcript_df = result_to_dataframe(result)
transcript_df.to_csv(file_name + "_transcipt.csv")

def diarization_to_dataframe(diarization):
    """
    Convert a diarization object into a pandas DataFrame.

    Args:
        diarization (object): A diarization object that contains information about segments and speakers.

    Returns:
        pandas.DataFrame: A DataFrame with columns 'start', 'end', and 'speaker', containing the diarization information.

    Example:
        diarization = ...  # diarization object
        df = diarization_to_dataframe(diarization)
        print(df)

        Output:
           start   end speaker
        0   0.00  10.00       A
        1  10.00  20.00       B
        2  20.00  30.00       A
    """
    df = pd.DataFrame(columns=['start', 'end', 'speaker'])

    for segment, _, speaker in diarization.itertracks(yield_label=True):
        df = df._append({
            'start': round(segment.start, 2),
            'end': round(segment.end, 2),
            'speaker': speaker}, 
            ignore_index=True)

    return df

diarization_df = diarization_to_dataframe(diarization)
diarization_df.to_csv(file_name + "_diarization.csv")

def merge_logic(transcript_df: pd.DataFrame, diarization_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the transcript and diarization dataframes based on the values of the 'start' and 'end' columns.

    Args:
        transcript_df (pd.DataFrame): A pandas DataFrame representing the transcript data. It should have columns named 'start', 'end', and 'text'.
        diarization_df (pd.DataFrame): A pandas DataFrame representing the diarization data. It should have columns named 'start' and 'speaker'.

    Returns:
        pd.DataFrame: A pandas DataFrame representing the merged data from `transcript_df` and `diarization_df`. The merged DataFrame has columns 'start', 'end', 'text', and 'speaker'. Each row in the merged DataFrame corresponds to a row in `transcript_df`, and the 'speaker' value is assigned based on the 'start' and 'end' values of the rows in `diarization_df`.
    """
    merge_df = pd.DataFrame(columns=['start', 'end', 'text', 'speaker'])

    diarization_index = 0
    speaker = diarization_df.iloc[diarization_index]['speaker']

    for index, line in transcript_df.iterrows():
        start = line['start']
        end = line['end']
        text = line['text']

        while diarization_index < len(diarization_df) - 1 and end > diarization_df.iloc[diarization_index + 1]['start']:
            diarization_index += 1
            speaker = diarization_df.iloc[diarization_index]['speaker']

        merge_df = merge_df._append({'start': start, 'end': end, 'text': text, 'speaker': speaker}, ignore_index=True)

    return merge_df

# Read CSV files
transcript_df = pd.read_csv(file_name + "_transcipt.csv")
diarization_df = pd.read_csv(file_name + "_diarization.csv")

# calling the merge_logic function
merge_df = merge_logic(transcript_df, diarization_df)

# Saving the merged dataframe to a CSV file
merge_df.to_csv(file_name + "_output_merge.csv")

### Create SRT file ###

srt_df = pd.read_csv(file_name + "_output_merge.csv", encoding='utf-8')

def seconds_to_time(seconds):
    """
    Converts a given number of seconds into the SRT time format (HH:MM:SS,MS).

    Args:
        seconds (int): The number of seconds to be converted into the SRT time format.

    Returns:
        str: The given number of seconds converted into the SRT time format.
    
    Example:
        time = seconds_to_time(3661)
        print(time)
        # Output: 01:01:01,000
    """
    milliseconds = int(seconds * 1000)
    seconds, milliseconds = divmod(milliseconds, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

# Generate SRT content
srt_content = ""
for i in range(srt_df.shape[0]):
    start_time = seconds_to_time((srt_df.loc[i, "start"]))
    end_time = seconds_to_time((srt_df.loc[i, "end"]))
    text = srt_df.loc[i, "speaker"] + ": " + srt_df.loc[i, "text"]
    srt_content += f"{i+1}\n{srt.srt_timestamp_to_timedelta(start_time)} --> {srt.srt_timestamp_to_timedelta(end_time)}\n{text}\n\n"

# SRT file
with open(file_name + ".srt", "w", encoding="utf-8") as f:
    f.write(srt_content)

### Create SRT file ###

### Making SUM file, wit hspeaker and the text ###

summ_df = pd.read_csv(file_name + "_output_merge.csv", encoding='utf-8')

sum_content = ""
for i in range(summ_df.shape[0]):
    text = summ_df.loc[i, "speaker"] + ": " + summ_df.loc[i, "text"]
    sum_content += text + "\n"

with open(file_name + ".txt", "w", encoding="utf-8") as f:
    f.write(sum_content)

### Making SUM file, wit hspeaker and the text ###
    
# print the result
with open(file_name + "_output.txt", "w", encoding="utf-8") as f:
        f.write(get_transcription(result))