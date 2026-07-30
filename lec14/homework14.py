import gtts, speech_recognition, librosa, soundfile

def synthesize(text, lang, filename):
    '''
    Use gtts.gTTS(text=text, lang=lang) to synthesize speech, then write it to filename.
    
    @params:
    text (str) - the text you want to synthesize
    lang (str) - the language in which you want to synthesize it
    filename (str) - the filename in which it should be saved
    '''
    # raise RuntimeError("You need to write this!")
    gtts.gTTS(text=text, lang=lang).save(filename)

def make_a_corpus(texts, languages, filenames):
    '''
    Create many speech files, and check their content using SpeechRecognition.
    The output files should be created as MP3, then converted to WAV, then recognized.

    @param:
    texts - a list of the texts you want to synthesize
    languages - a list of their languages
    filenames - a list of their root filenames, without the ".mp3" ending

    @return:
    recognized_texts - list of the strings that were recognized from each file
    '''
    #raise RuntimeError("You need to write this!")
    r = speech_recognition.Recognizer()
    recognized_texts = []
    
    for tup in zip(texts, languages, filenames):
        synthesize(tup[0], tup[1], tup[2]+'.mp3')
        data, samplerate = librosa.load(tup[2] + '.mp3')
        soundfile.write(tup[2]+'.wav', data, samplerate)
        with speech_recognition.AudioFile(tup[2]+ '.wav') as source:
            audio = r.record(source)
            text = r.recognize_google(audio, language=tup[1])
            recognized_texts.append(text)
    return recognized_texts
