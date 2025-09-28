# Character-Level RNN Text Generator

This project is a **character-level RNN text generator** built using TensorFlow.  
This project was made for **educational purposes**.  
It can generate text one character at a time, trained on either the **Tiny Shakespeare dataset** or a **Dinosaur names dataset** from Karpathy.

---

## Features
- Train on two datasets: **Shakespeare text** or **Dinosaur names**.
- Generates text character by character using an RNN.
- Supports **temperature sampling** for controlling randomness:
  - Low temperature → more predictable text
  - High temperature → more creative text
- Easy to **retrain** by setting `TRAINING = True`.
- Adjustable parameters:
  - Number of **epochs**
  - **Batch size**
  - **Sequence length**
  - **Temperature**
  - Dataset selection (`shakespeare` or `dino`)

---

## Limitations
- Uses **character-level RNN** with **simple LSTM layers**, so it may not produce very high-quality or coherent text.  
- Designed for **small datasets**, so results may be limited with larger or more complex corpora.  
- Text generation quality depends heavily on **temperature** and sequence length.

---

## Usage

### 1. Select Dataset
```python
DATASET = "shakespeare"  # or "dino"
```

### 2. Set Training Options
```python
TRAINING = True          # True to train, False to generate text
EPOCHS = 100             # Number of training epochs
temperature = 0.5        # Lower = more predictable, higher = more random
```

### 3. Run the Script
```bash
python main.py
```

### 4. Behavior
- `TRAINING = True` → trains the model on the selected dataset  
- `TRAINING = False` → generates text using saved model weights  

### 5. Example Output
Enter seed text to generate dino text: tyr

--- Generated text ---

tyraurus
Sylos
Amiaus

## Requirements
- Install Python packages using:
```bash
pip install -r requirements.txt
```

Typical packages:
```
tensorflow>=2.12
numpy
```

---

## License
- MIT License