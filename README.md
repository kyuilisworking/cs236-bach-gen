# Bach-Gen

The important files to look at are:
1. `lstm_vae.py`
2. `v3_lstm.py`
3. `train_lstm_vae.py`
4. `preprocess_data.py`
5. `load_data.py`

The data is uploaded separately to Google Drive.

`preproccess_data` creates 32-note sequences of 16th notes from MIDI files of Bach's works.

v3_lstm implements a version of MusicVAE. It implements the LSTM encoder as well as a categorical LSTM decoder and a hierarchical LSTM decoder (needs debugging).