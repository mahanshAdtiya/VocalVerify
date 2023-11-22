from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import IPython
import wave
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score
import librosa
from io import BytesIO
import base64

app = Flask(__name__)

data=pd.read_csv("../DATASET-balanced.csv")
X = data.iloc[:,:-1]
Y = data.iloc[:,-1]

lb = preprocessing.LabelBinarizer()
lb.fit(Y)
Y = lb.transform(Y)
Y = Y.ravel()

model = RandomForestClassifier(n_estimators=50, random_state=1)

from sklearn.model_selection import KFold
kf = KFold(n_splits=5,  shuffle=True, random_state=1)

acc_score = []
prec_score = []
rec_score = []
f1s = []
MCCs = []
ROCareas = []

start = time.time()
for train_index , test_index in kf.split(X):
    X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
    Y_train , Y_test = Y[train_index] , Y[test_index]

    model.fit(X_train,Y_train)
    pred_values = model.predict(X_test)

    acc = accuracy_score(pred_values , Y_test)
    acc_score.append(acc)

    prec = precision_score(Y_test , pred_values, average="binary", pos_label=1)
    prec_score.append(prec)

    rec = recall_score(Y_test , pred_values, average="binary", pos_label=1)
    rec_score.append(rec)

    f1 = f1_score(Y_test , pred_values, average="binary", pos_label=1)
    f1s.append(f1)

    mcc = matthews_corrcoef(Y_test , pred_values)
    MCCs.append(mcc)

    roc = roc_auc_score(Y_test , pred_values)
    ROCareas.append(roc)

end = time.time()
timeTaken = (end - start)
time_taken = "Model trained in: " + str( round(timeTaken, 2) ) + " seconds."
accuracy="Accuracy: " + str( round(np.mean(acc_score)*100, 3) ) + "% (" + str( round(np.std(acc_score)*100, 3) ) + ")"
precision="Precision: " + str( round(np.mean(prec_score), 3) ) + " (" + str( round(np.std(prec_score), 3) ) + ")"
recall="Recall: " + str( round(np.mean(rec_score), 3) ) + " (" + str( round(np.std(rec_score), 3) ) + ")"
f1="F1-Score: " + str( round(np.mean(f1s), 3) ) + " (" + str( round(np.std(f1s), 3) ) + ")"
mcc="MCC: " + str( round(np.mean(MCCs), 3) ) + " (" + str( round(np.std(MCCs), 3) ) + ")"
roc_au="ROC AUC: " + str( round(np.mean(ROCareas), 3) ) + " (" + str( round(np.std(ROCareas), 3) ) + ")"

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

name=""
filepath=""

@app.route('/intro')
def intro():
    margot_robbie_speech = wave.open('../AUDIO/REAL/margot-original.wav', 'rb')
    margot_robbie_fake_speech = wave.open('../AUDIO/FAKE/margot-to-ryan.wav', 'rb')

    # print(margot_robbie_speech)
    # print("hi-2")
    sample_freq = margot_robbie_speech.getframerate()
    n_samples = margot_robbie_speech.getnframes()
    t_audio = n_samples/sample_freq
    n_channels = margot_robbie_speech.getnchannels()

    sample_rate="The samping rate of the audio file is " + str(sample_freq) + "Hz, or " + str(sample_freq/1000) + "kHz"
    frames="The audio contains a total of " + str(n_samples) + " frames or samples"
    length="The length of the audio file is " + str(t_audio) + " seconds"
    channels="The audio file has " + str(n_channels) + " channels."

    signal_wave = margot_robbie_speech.readframes(n_samples)
    signal_array = np.frombuffer(signal_wave, dtype=np.int16)


    samples="The signal contains a total of " + str(signal_array.shape[0]) + " samples."
    line="If this value is greater than " + str(n_samples) + " it is due to there being multiple channels." + "\n" + "E.g. - Samples * Channels = " + str(n_samples*n_channels)

    # Split the channels
    l_channel = signal_array[0::2]
    r_channel = signal_array[1::2]

    timestamps = np.linspace(0, n_samples/sample_freq, num=n_samples)
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, l_channel)
    plt.title('Left Channel')
    plt.ylabel('Signal Value')
    plt.xlabel('Time (s)')
    plt.xlim(0, t_audio)
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    img1 = base64.b64encode(img_buf.read()).decode('utf-8')

    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, r_channel)
    plt.title('Right Channel')
    plt.ylabel('Signal Value')
    plt.xlabel('Time (s)')
    plt.xlim(0, t_audio)
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    img2 = base64.b64encode(img_buf.read()).decode('utf-8')

    plt.figure(figsize=(10, 5))
    plt.specgram(l_channel, Fs=sample_freq, vmin=-20, vmax=50)
    plt.title('Left Channel')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.xlim(0, t_audio)
    plt.colorbar()
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    img3 = base64.b64encode(img_buf.read()).decode('utf-8')

    plt.figure(figsize=(10, 5))
    plt.specgram(r_channel, Fs=sample_freq, vmin=-20, vmax=50)
    plt.title('Right Channel')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.xlim(0, t_audio)
    plt.colorbar()
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    img4 = base64.b64encode(img_buf.read()).decode('utf-8')

    sample_freq_fake = margot_robbie_fake_speech.getframerate()
    n_samples_fake = margot_robbie_fake_speech.getnframes()
    t_audio_fake = n_samples/sample_freq
    n_channels_fake = margot_robbie_fake_speech.getnchannels()

    signal_wave_fake = margot_robbie_fake_speech.readframes(n_samples_fake)
    signal_array_fake = np.frombuffer(signal_wave_fake, dtype=np.int16)


    l_channel_fake = signal_array_fake[0::2]

    plt.figure(figsize=(10, 5))
    plt.specgram(l_channel, Fs=sample_freq, vmin=-20, vmax=50)
    plt.title('Left Channel - REAL MARGOT ROBBIE')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.xlim(0, t_audio)
    plt.colorbar()
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    img5 = base64.b64encode(img_buf.read()).decode('utf-8')

    plt.figure(figsize=(10, 5))
    plt.specgram(l_channel_fake, Fs=sample_freq_fake, vmin=-20, vmax=50)
    plt.title('Left Channel - FAKE MARGOT ROBBIE (Ryan Gosling)')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.xlim(0, t_audio_fake)
    plt.colorbar()
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    img6 = base64.b64encode(img_buf.read()).decode('utf-8')

    return {"intro" : [sample_rate,frames,length,channels,samples,line],"image":[img1,img2,img3,img4,img5,img6],"result":[accuracy,precision,recall,f1,mcc,roc_au]}

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'audioFile' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        audio_file = request.files['audioFile']

        if audio_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if audio_file and allowed_file(audio_file.filename):
            # Save the file to the uploads folder
            filename = secure_filename(audio_file.filename)
            global name 
            name = filename
            global filepath 
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            audio_file.save(filepath)

            return jsonify({'message': 'File uploaded successfully'})
        else:
            return jsonify({'error': 'Invalid file format'}), 400

    except Exception as e:
        return jsonify({'error': f'Error uploading file: {str(e)}'}), 500

@app.route('/test')
def test():
    
    test = wave.open(filepath, 'rb')

    # print(margot_robbie_speech)
    # print("hi-2")
    sample_freq = test.getframerate()
    n_samples = test.getnframes()
    t_audio = n_samples/sample_freq
    n_channels = test.getnchannels()

    sample_rate="The samping rate of the audio file is " + str(sample_freq) + "Hz, or " + str(sample_freq/1000) + "kHz"
    frames="The audio contains a total of " + str(n_samples) + " frames or samples"
    length="The length of the audio file is " + str(t_audio) + " seconds"
    channels="The audio file has " + str(n_channels) + " channels."

    signal_wave = test.readframes(n_samples)
    signal_array = np.frombuffer(signal_wave, dtype=np.int16)


    samples="The signal contains a total of " + str(signal_array.shape[0]) + " samples."
    line="If this value is greater than " + str(n_samples) + " it is due to there being multiple channels." + "\n" + "E.g. - Samples * Channels = " + str(n_samples*n_channels)

    # Split the channels
    l_channel = signal_array[0::2]
    r_channel = signal_array[1::2]

    timestamps = np.linspace(0, n_samples/sample_freq, num=n_samples)
    
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, l_channel)
    plt.title('Left Channel')
    plt.ylabel('Signal Value')
    plt.xlabel('Time (s)')
    plt.xlim(0, t_audio)
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    img1 = base64.b64encode(img_buf.read()).decode('utf-8')

    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, r_channel)
    plt.title('Right Channel')
    plt.ylabel('Signal Value')
    plt.xlabel('Time (s)')
    plt.xlim(0, t_audio)
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    img2 = base64.b64encode(img_buf.read()).decode('utf-8')

    plt.figure(figsize=(10, 5))
    plt.specgram(l_channel, Fs=sample_freq, vmin=-20, vmax=50)
    plt.title('Left Channel')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.xlim(0, t_audio)
    plt.colorbar()
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    img3 = base64.b64encode(img_buf.read()).decode('utf-8')

    plt.figure(figsize=(10, 5))
    plt.specgram(r_channel, Fs=sample_freq, vmin=-20, vmax=50)
    plt.title('Right Channel')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.xlim(0, t_audio)
    plt.colorbar()
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    img4 = base64.b64encode(img_buf.read()).decode('utf-8')

    print(name)
    testing_audio_path = filepath  # Replace this with the actual path to your testing audio file
    testing_audio, sr = librosa.load(testing_audio_path, sr=None)

    chroma_stft = np.mean(librosa.feature.chroma_stft(y=testing_audio, sr=sr))
    rms = np.mean(librosa.feature.rms(y=testing_audio))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=testing_audio, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=testing_audio, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=testing_audio, sr=sr))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=testing_audio))

    mfccs = librosa.feature.mfcc(y=testing_audio, sr=sr, n_mfcc=20)
    mfccs_mean = np.mean(mfccs, axis=1)
    feature_vector = np.hstack((chroma_stft, rms, spectral_centroid, spectral_bandwidth, rolloff, zero_crossing_rate, mfccs_mean))

    if feature_vector.shape[0] != 26:
        print("Error: Incorrect number of features extracted from the testing audio")
    else:
        # Predict using the trained model
        prediction = model.predict([feature_vector])

        # Convert the prediction to the corresponding label
        predicted_label = "REAL" if prediction[0] == 1 else "FAKE"
        predicted_label="Predicted label for the testing audio: " + predicted_label

    print("yaha toh m aagya hu")
    return {"test" : [sample_rate,frames,length,channels,samples,line],"image":[img1,img2,img3,img4],"result":[predicted_label]}


if __name__ == '__main__':
    app.run(debug=True) 