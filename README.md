# Music Generation using LSTM

## Overview
This project focuses on generating music using Long Short-Term Memory (LSTM) neural networks. The model is trained on the **MusicNet dataset**, which contains a collection of classical music pieces annotated with notes and timing information. The goal is to develop an LSTM-based deep learning model capable of composing music sequences.

## Dataset: MusicNet 
**MusicNet** is a large-scale dataset that includes:
- 330 classical music recordings
- Over **1 million** annotated notes with detailed pitch, instrument, and timing information
- Data in WAV format with corresponding labels for supervised learning

### Dataset Source:
You can download the dataset from Kaggle: [MusicNet Dataset](https://www.kaggle.com/c/musicnet)

## Project Dependencies
To run this project, install the required Python libraries:
```bash
pip install torch torchvision torchaudio tqdm numpy pandas matplotlib
```

## Model Architecture
The project utilizes a **Recurrent Neural Network (RNN)** with LSTM layers to process sequential music data.
### Key Components:
1. **LSTM Layers:** Captures temporal dependencies in music sequences.
2. **Fully Connected Layers:** Transforms the LSTM outputs into note predictions.
3. **Loss Function:** Cross-entropy loss for categorical classification.
4. **Optimizer:** Adam optimizer for efficient gradient descent.
![image](https://github.com/user-attachments/assets/b722daa6-db18-42be-b681-d57eb062b6be)


## Training Process
1. **Data Preprocessing:**
   - Convert audio data into a sequence of musical notes.
   - Normalize and reshape input sequences.
   
2. **Training:**
   - Train the LSTM model using labeled sequences from MusicNet.
   - Track loss and accuracy using validation data.
   
3. **Evaluation:**
   - Generate new music sequences based on the learned patterns.

## Running the Model
To train the model, execute the following command in a Jupyter Notebook or Python script:
```python
python music-generation-using-lstm.ipynb
```

## Results and Output
- The trained LSTM model generates sequences of musical notes.
- The output can be converted into MIDI format for playback.
- Visualizations include training loss graphs and note distribution plots.

## Future Improvements
- Implement **Bidirectional LSTMs** for improved performance.
- Use **Transformer models** for enhanced long-range dependencies.
- Experiment with different dataset augmentation techniques.

## References
- Kaggle MusicNet Dataset: https://www.kaggle.com/c/musicnet
- PyTorch Documentation: https://pytorch.org/docs/stable/index.html
