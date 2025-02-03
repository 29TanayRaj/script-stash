from youtube_transcript_api import YouTubeTranscriptApi
import re 

# Function to fetch transcript of a youtube video.
def get_transcript(vid_code):

    """
    Function to get trancript of a youtube video
    
    Parameters-
    vid_code: str type, code form the youtube video. e.g In this URL 'https://www.youtube.com/watch?v=dQw4w9WgXcQ', 'dQw4w9WgXcQ' is the video code.

    Returns transcipt of the video in a str format
    """

    # Getting the subtitles from the video
    vid_sub = YouTubeTranscriptApi.get_transcript(vid_code)

    # Vid_sub is a list of dicts, we only need text from each dict

    transcript = ''
    for dict in vid_sub:
        transcript = transcript + ' '+ dict['text']

    # removing the new line '\n' symbol

    transcript = re.sub("\n",'',transcript)

    return transcript
