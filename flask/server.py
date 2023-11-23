import json
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
import warnings

warnings.filterwarnings("ignore")

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
accuracy="Accuracy: " + str( round(np.mean(acc_score)*100, 3) ) + "%"
precision="Precision: " + str( round(np.mean(prec_score), 3) )
recall="Recall: " + str( round(np.mean(rec_score), 3) )
f1="F1-Score: " + str( round(np.mean(f1s), 3) )
mcc="MCC: " + str( round(np.mean(MCCs), 3) )
roc_au="ROC AUC: " + str( round(np.mean(ROCareas), 3) )

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

name=""
filepath=""

@app.route('/intro')
def intro():
    return {"intro" : [],"image":[],"result":[accuracy,precision,recall,f1,mcc,roc_au]}

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

def extract_features(file_path, segment_length, file_name):
    try:
        y, sr = librosa.load(file_path)
        num_segments = int(np.ceil(len(y) / float(segment_length * sr)))

        features = []

        for i in range(num_segments):
            start_frame = i * segment_length * sr
            end_frame = min(len(y), (i + 1) * segment_length * sr)

            y_segment = y[start_frame:end_frame]

            chroma_stft = np.mean(librosa.feature.chroma_stft(y=y_segment, sr=sr))
            rms = np.mean(librosa.feature.rms(y=y_segment))
            spec_cent = np.mean(librosa.feature.spectral_centroid(y=y_segment, sr=sr))
            spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y_segment, sr=sr))
            rolloff = np.mean(librosa.feature.spectral_rolloff(y=y_segment, sr=sr))
            zcr = np.mean(librosa.feature.zero_crossing_rate(y_segment))
            mfccs = librosa.feature.mfcc(y=y_segment, sr=sr)
            mfccs_mean = np.mean(mfccs, axis=1)

            features.append([chroma_stft, rms, spec_cent, spec_bw, rolloff, zcr, *mfccs_mean])

        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

@app.route('/test')
def test():
    
    test = wave.open(filepath, 'rb')

    # print(margot_robbie_speech)
    print("hi-2")
    sample_freq = test.getframerate()
    n_samples = test.getnframes()
    t_audio = n_samples/sample_freq
    n_channels = test.getnchannels()

    sample_rate="The samping rate of the audio file is " + str(sample_freq) + "Hz, or " + str(sample_freq/1000) + "kHz"
    framess="The audio contains a total of " + str(n_samples) + " frames or samples"
    length="The length of the audio file is " + str(t_audio) + " seconds"
    channels="The audio file has " + str(n_channels) + " channels."

    signal_wave = test.readframes(n_samples)
    signal_array = np.frombuffer(signal_wave, dtype=np.int16)

    print("hi-3")
    samples="The signal contains a total of " + str(signal_array.shape[0]) + " samples."
<<<<<<< HEAD
    # line="If this value is greater than " + str(n_samples) + " it is due to there being multiple channels." + "\n" + "E.g. - Samples * Channels = " + str(n_samples*n_channels)
   
=======
    
>>>>>>> a181ae1124913cb4bd1b81e8a3fe67767d37204f
    print("hi-4")
    y, sr = librosa.load(filepath)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    db_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(db_spectrogram, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    img1 = base64.b64encode(img_buf.read()).decode('utf-8')
    print("hi-5")

    with wave.open(filepath, 'rb') as wf:
        num_frames = wf.getnframes()
        frames = wf.readframes(num_frames)
        signal = np.frombuffer(frames, dtype=np.int16)
        frame_rate = wf.getframerate()

        # Calculate time array based on the duration of the audio
        duration = num_frames / frame_rate
        time = np.linspace(0, duration, len(signal))

        plt.figure(figsize=(10, 4))
        plt.plot(time, signal / 10000, linewidth=0.5)

        plt.title('Waveform of {}'.format(filepath))
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)

        img_buf = BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        img2 = base64.b64encode(img_buf.read()).decode('utf-8')


    print(name)
    testing_audio_path = filepath  # Replace this with the actual path to your testing audio file
    # testing_audio, sr = librosa.load(testing_audio_path, sr=None)

    # chroma_stft = np.mean(librosa.feature.chroma_stft(y=testing_audio, sr=sr))
    # rms = np.mean(librosa.feature.rms(y=testing_audio))
    # spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=testing_audio, sr=sr))
    # spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=testing_audio, sr=sr))
    # rolloff = np.mean(librosa.feature.spectral_rolloff(y=testing_audio, sr=sr))
    # zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=testing_audio))

    # mfccs = librosa.feature.mfcc(y=testing_audio, sr=sr, n_mfcc=20)
    # mfccs_mean = np.mean(mfccs, axis=1)
    # feature_vector = np.hstack((chroma_stft, rms, spectral_centroid, spectral_bandwidth, rolloff, zero_crossing_rate, mfccs_mean))

    # if feature_vector.shape[0] != 26:
    #     print("Error: Incorrect number of features extracted from the testing audio")
    # else:
    #     # Predict using the trained model
    #     prediction = model.predict([feature_vector])

    #     # Convert the prediction to the corresponding label
    #     predicted_label = "REAL" if prediction[0] == 1 else "FAKE"
    #     predicted_label="Predicted label for the testing audio: " + predicted_label

    testing_audio_features = extract_features(testing_audio_path, 5, "testing_audio")

    feature_vector = np.mean(testing_audio_features, axis=0)

    if len(feature_vector) != 26:
        print("Error: Incorrect number of features extracted from the testing audio")
    else:

        prediction = model.predict([feature_vector])

        predicted_label = "REAL" if prediction[0] == 1 else "FAKE"

        print("Predicted label for the testing audio: " + predicted_label)

    print("yaha toh m aagya hu")
    return {"test" : [sample_rate,framess,length,channels,samples],"image":[img1,img2],"result":[predicted_label]}
    # return json_result


if __name__ == '__main__':
    app.run(debug=True) 
